import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image


_FONT_CACHE = {}


def _get_cached_font(font_size):
    key = int(font_size)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]

    font_list = [
        "simsun.ttc",
        "simhei.ttf",
        "simkai.ttf",
        "msyh.ttc",
        "STKAITI.TTF",
        "STZHONGS.TTF",
        "arial.ttf",
    ]

    font = None
    for font_name in font_list:
        try:
            font = ImageFont.truetype(font_name, key)
            break
        except IOError:
            continue

    if font is None:
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

    _FONT_CACHE[key] = font
    return font


def draw_chinese_texts(img, items):
    if img is None or not items:
        return img

    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    for text, position, font_size, color in items:
        pil_color = (color[2], color[1], color[0])
        font = _get_cached_font(font_size)
        if font is None:
            cv_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            cv2.putText(
                cv_img,
                str(text),
                position,
                cv2.FONT_HERSHEY_SIMPLEX,
                float(font_size) / 30.0,
                color,
                2,
                cv2.LINE_AA,
            )
            img_pil = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            continue

        draw.text(position, str(text), font=font, fill=pil_color)

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def draw_chinese_text(img, text, position, font_size=24, color=(255, 0, 0)):
    return draw_chinese_texts(img, [(text, position, font_size, color)])

