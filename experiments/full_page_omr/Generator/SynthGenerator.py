import re
import os
import sys
import tempfile
from contextlib import contextmanager

import cv2

import verovio
import random
from datasets import load_dataset

from PIL import Image, ImageOps
from wand.image import Image as IMG

import numpy as np
import xml.etree.ElementTree as ET

from rich import progress
from cairosvg import svg2png

import names
from loguru import logger
from wonderwords import RandomSentence

from ..tokenization import validate_tokenization_mode


class SyntheticScoreRenderError(RuntimeError):
    """Raised when a generated score cannot be rendered safely."""


class SyntheticScoreGenerationError(RuntimeError):
    """Raised when a complete sample exhausts its render attempts."""


@contextmanager
def _capture_native_stderr():
    """Capture native-library stderr so recoverable parser errors can be rejected."""
    captured = [""]
    stderr_fd = 2
    try:
        sys.stderr.flush()
        saved_stderr_fd = os.dup(stderr_fd)
    except (AttributeError, OSError, ValueError):
        yield captured
        return

    with tempfile.TemporaryFile(mode="w+b") as buffer:
        try:
            os.dup2(buffer.fileno(), stderr_fd)
            yield captured
        finally:
            sys.stderr.flush()
            os.dup2(saved_stderr_fd, stderr_fd)
            os.close(saved_stderr_fd)
            buffer.seek(0)
            captured[0] = buffer.read().decode(errors="replace")


def _native_stderr_has_error(output):
    return any(line.lstrip().startswith(("Error:", "[Error]")) for line in output.splitlines())


def _native_error_summary(output):
    return next(
        (line.strip() for line in output.splitlines() if line.strip()),
        "native parser error",
    )


def clean_kern(krn, avoid_tokens=['*Xped', '*staff1', '*staff2', '*tremolo', 
                                  '*ped', '*Xtuplet', '*tuplet', "*Xtremolo", 
                                  '*cue', '*Xcue', '*rscale:1/2', '*rscale:1', 
                                  '*kcancel', '*below']):
    krn = krn.split('\n')
    newkrn = []
    # Remove the lines that contain the avoid tokens
    for idx, line in enumerate(krn):
        if not any([token in line.split('\t') for token in avoid_tokens]):
            #If all the tokens of the line are not '*'
            if not all([token == '*' for token in line.split('\t')]):
                newkrn.append(line.replace("\n", ""))
                
    return "\n".join(newkrn)

#@memory.cache
def parse_kern(krn: str) -> str:
    krn = clean_kern(krn)
    krn = krn.replace(" ", " <s> ")
    krn = krn.replace("\t", " <t> ")
    krn = krn.replace("\n", " <b> ")
    krn = krn.replace("·/", "")
    krn = krn.replace("·\\", "")
        
    krn = krn.split(" ")[4:]
    krn = [re.sub(r'(?<=\=)\d+', '', token) for token in krn]
    
    return " ".join(krn)

def load_from_files_list(dataset_ref: list, split:str="train") -> list:
    return [parse_kern(content) for content in progress.track(load_dataset(dataset_ref, split=split)["transcription"])]

def rfloat(start, end):
    return round(random.uniform(start, end), 2)

def rint(start, end):
    return random.randint(start, end)        

class VerovioGenerator():
    def __init__(self, sources: list, split="train", tokenization_mode='bekern', max_render_attempts=20):
        if max_render_attempts < 1:
            raise ValueError("max_render_attempts must be at least 1")

        tokenization_mode = validate_tokenization_mode(tokenization_mode)
        self.beat_db = self.load_beats(sources, split=split)
        verovio.enableLog(verovio.LOG_OFF)
        self.tk = verovio.toolkit()

        self.tokenization_mode = tokenization_mode
        self.max_render_attempts = max_render_attempts
        self.render_attempts = 0
        self.render_failures = 0
        self.title_generator = RandomSentence()
        self.textures = [os.path.join("Generator/paper_textures", f) for f in os.listdir("Generator/paper_textures") if os.path.isfile(os.path.join("Generator/paper_textures", f))]

        
    def load_beats(self, sources: list, split:str):
        sequences = load_from_files_list(sources, split=split)
        beats = {}
        for sequence in sequences:
            if sequence.count('*-') == 2:
                lines_sequence = sequence.replace('<b>', '\n').split('\n')
                for line in lines_sequence:
                    if "*M" in line:
                        beat_marker = re.search(r'\*M\S*', line)
                        beats.setdefault(beat_marker.group(), []).append(sequence)
        
        #Remove all the keys and elements that have less than 6 elements
        keys = list(beats.keys())
        for key in keys:
            if len(beats[key]) < 6:
                del beats[key]
        
        return beats

    def count_class_occurrences(self, svg_file, class_name):
        root = ET.fromstring(svg_file)
        count = 0
        # Define the namespace for SVG, if necessary
        # svg_namespace = '{http://www.w3.org/2000/svg}'
        # Traverse the SVG tree to count occurrences of the class
        for element in root.iter():
            # You can use the following line if your SVG uses a namespace:
            # if class_name in element.get('class', '').split():
            if class_name in element.get('class', ''):
                count += 1

        return count
    
    def find_image_cut(self, sample):
        # Get the height of the image
        height, _ = sample.shape[:2]

        # Iterate through the image array from the bottom to the top
        for y in range(height - 1, -1, -1):
            if [0, 0, 0] in sample[y]:
                return y

        # If no black pixel is found, return None
        return None
    
    def render(self, music_sequence):
        self.render_attempts += 1
        with _capture_native_stderr() as native_stderr:
            loaded = self.tk.loadData(music_sequence)

        if not loaded or _native_stderr_has_error(native_stderr[0]):
            reason = (
                _native_error_summary(native_stderr[0])
                if native_stderr[0]
                else "Verovio rejected the generated Humdrum score"
            )
            self._raise_render_error(reason)
        elif native_stderr[0]:
            sys.stderr.write(native_stderr[0])
            sys.stderr.flush()

        options_set = self.tk.setOptions({"pageWidth": 2100, "footer": 'none',
                                          'barLineWidth': rfloat(0.3, 0.8), 'beamMaxSlope': rfloat(10,20),
                                          'staffLineWidth': rfloat(0.1, 0.3), 'spacingStaff': rfloat(1, 12)})
        if not options_set:
            self._raise_render_error("Verovio rejected the rendering options")
        if self.tk.getPageCount() < 1:
            self._raise_render_error("Verovio produced no pages for the generated score")

        svg = self.tk.renderToSVG()
        if not svg:
            self._raise_render_error("Verovio returned an empty SVG")
        try:
            ET.fromstring(svg)
        except ET.ParseError as exc:
            self._raise_render_error("Verovio returned malformed SVG", cause=exc)

        svg = svg.replace("overflow=\"inherit\"", "overflow=\"visible\"")
        return svg

    def _raise_render_error(self, reason, cause=None):
        self.render_failures += 1
        if self.render_failures == 1 or self.render_failures % 25 == 0:
            logger.warning(
                "Rejected invalid synthetic score: {} (render failures: {}/{})",
                reason,
                self.render_failures,
                self.render_attempts,
            )
        error = SyntheticScoreRenderError(reason)
        if cause is None:
            raise error
        raise error from cause
    
    def convert_to_png(self, svg_file, cut=False):
        pngfile = svg2png(bytestring=svg_file, background_color='white')
        pngfile = cv2.imdecode(np.frombuffer(pngfile, np.uint8), -1)
        if pngfile is None or pngfile.size == 0:
            self._raise_render_error("The generated SVG could not be decoded as an image")
        if cut:
            cut_height = self.find_image_cut(pngfile)
            if cut_height is not None:
                pngfile = pngfile[:cut_height + 10, :]
        
        return pngfile
    
    def inkify_image(self, sample):
        image = IMG.from_array(np.array(sample))
        paint = rfloat(0, 1)
        image.oil_paint(paint)
        
        return Image.fromarray(np.array(image))
    
    def filter_system_continuation(self, system, cut_end=True):
        if cut_end:
            system = system[:-5]
        
        system = " ".join(system).split(" <b> ")
        ignored_indices = []
        for idx, line in enumerate(system):
            if any(token in line for token in ['*clef', '*k', '*M', '*met']):
                ignored_indices.append(idx)
            if "=" in line:
                break
        system = [line for idx, line in enumerate(system) if idx not in ignored_indices]

        system = " <b> ".join(system).split(' ')
        return [token for token in system if token != '']
    
    def generate_music_system_image(self, reduce_ratio=0.5):
        for _ in range(self.max_render_attempts):
            beat = random.choice(list(self.beat_db.keys()))
            music_seq = random.choice(self.beat_db[beat])
            render_sequence = "**kern\t**kern\n" + music_seq.replace(' <b> ', '\n').replace(' <s> ', ' ').replace(' <t> ', '\t').replace('@', '').replace('·', '')
            try:
                image = self.render(render_sequence)
            except SyntheticScoreRenderError:
                continue

            num_systems = self.count_class_occurrences(svg_file=image, class_name='grpSym')
            if num_systems != 1:
                continue

            try:
                x = self.convert_to_png(image, cut=True)
            except SyntheticScoreRenderError:
                continue
            break
        else:
            raise SyntheticScoreGenerationError(
                f"Unable to generate a valid single-system score after {self.max_render_attempts} attempts"
            )

        x = cv2.cvtColor(np.array(x), cv2.COLOR_BGR2RGB)
        width = int(np.ceil(x.shape[1] * reduce_ratio))
        height = int(np.ceil(x.shape[0] * reduce_ratio))
        x = cv2.resize(x, (width, height))
        
        gt_sequence = ""
        
        if self.tokenization_mode == "kern":
            gt_sequence = "".join(music_seq).replace("<s>", " <s> ").replace("<b>", " <b> ").replace("<t>", " <t> ").replace("·", "").replace('@', '').split(" ")
        
        if self.tokenization_mode == "ekern":
            gt_sequence = "".join(music_seq).replace("<s>", " <s> ").replace("<b>", " <b> ").replace("<t>", " <t> ").replace("·", " ").replace('@', '').split(" ")
        
        if self.tokenization_mode == "bekern":
            gt_sequence = "".join(music_seq).replace("<s>", " <s> ").replace("<b>", " <b> ").replace("<t>", " <t> ").replace("·", " ").replace("@", " ").split(" ")

        return x, ['<bos>'] + [token for token in gt_sequence if token != ''] + ['<eos>']

    def generate_full_page_score(self, max_systems=2, strict_systems=False, strict_height=False, 
                                 include_title=False, include_author=False, texturize_image=True, reduce_ratio=0.5):
        num_systems = max_systems
        eligible_beats = [beats for beats in self.beat_db.values() if len(beats) >= num_systems]
        if not eligible_beats:
            raise ValueError(f"No beat group contains {num_systems} source systems")

        for _ in range(self.max_render_attempts):
            beats = random.choice(eligible_beats)
            
            systems_to_compose = [system.split(" ") for system in random.sample(beats, num_systems)]
            complete_score = []
            if len(systems_to_compose) > 1:
                complete_score = systems_to_compose[0][:-5]
                for system in systems_to_compose[1:-1]:
                        complete_score += ["<b>"] + self.filter_system_continuation(system)
                complete_score += ["<b>"] + self.filter_system_continuation(systems_to_compose[-1], cut_end=False)
            else:
                complete_score = systems_to_compose[0]

            complete_score = [token for idx, token in enumerate(complete_score) if token != '<b>' or (idx > 0 and complete_score[idx - 1] != '<b>')]

            preseq = ""
            preseq += f"!!!OTL:{self.title_generator.sentence()}\n" if include_title else ""
            #preseq += f"!!!COM:{names.get_full_name()}\n" if include_author else ""
            render_sequence = preseq + "**kern\t**kern\n" + "".join(complete_score).replace('<b>', '\n').replace('<s>', ' ').replace('<t>', '\t').replace('@', '').replace('·', '')
            
            try:
                image = self.render(render_sequence)
            except SyntheticScoreRenderError:
                continue

            generated_systems = self.count_class_occurrences(svg_file=image, class_name='grpSym')
            if generated_systems < 1 or (strict_systems and generated_systems != num_systems):
                continue

            try:
                x = self.convert_to_png(image, cut=strict_height)
            except SyntheticScoreRenderError:
                continue
            break
        else:
            raise SyntheticScoreGenerationError(
                f"Unable to generate a valid {num_systems}-system score after "
                f"{self.max_render_attempts} attempts"
            )

        texture = Image.open(random.choice(self.textures))
        img_width, img_height = x.shape[1], x.shape[0]
        #Check if the texture is smaller than the image. If so, resize it to the image size
        if texture.size[0] < img_width or texture.size[1] < img_height:
            texture = texture.resize((img_width, img_height))
        
        x = self.inkify_image(x)
        x = np.array(x)
        music_image = Image.fromarray(x)
        left = random.randint(0, texture.size[0] - img_width)
        top = random.randint(0, texture.size[1] - img_height)
        texture = texture.crop((left, top, left + img_width, top + img_height))
        inverted_music_image = ImageOps.invert(music_image.convert('RGB'))
        mask = inverted_music_image.convert("L")
        texture.paste(music_image, mask=mask)
        x = texture#cv2.cvtColor(np.array(texture), cv2.COLOR_RGB2BGR)
        
        x = cv2.cvtColor(np.array(x), cv2.COLOR_BGR2RGB)
        
        width = int(np.ceil(x.shape[1] * reduce_ratio))
        height = int(np.ceil(x.shape[0] * reduce_ratio))
        x = cv2.resize(x, (width, height))
   
        gt_sequence = ""
        
        if self.tokenization_mode == "kern":
            gt_sequence = "".join(complete_score).replace("<s>", " <s> ").replace("<b>", " <b> ").replace("<t>", " <t> ").replace("·", "").replace('@', '').split(" ")
        
        if self.tokenization_mode == "ekern":
            gt_sequence = "".join(complete_score).replace("<s>", " <s> ").replace("<b>", " <b> ").replace("<t>", " <t> ").replace("·", " ").replace('@', '').split(" ")
        
        if self.tokenization_mode == "bekern":
            gt_sequence = "".join(complete_score).replace("<s>", " <s> ").replace("<b>", " <b> ").replace("<t>", " <t> ").replace("·", " ").replace("@", " ").split(" ")

        return x, ['<bos>'] + [token for token in gt_sequence if token != ''] + ['<eos>']
                
        
