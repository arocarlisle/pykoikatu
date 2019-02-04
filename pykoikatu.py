# A chara card contains two pngs (cover, head) and chara data.
# The chara data contains lstInfo and four lists (custom, coordinate,
#   parameter, status).
# The custom list contains three lists (face, body, hair).
# Each list contains several tokens. Data types of tokens are shown below.
# The file structure is really awful... Why don't they use a common pickler?

# TODO:
# Better parameter models
# Generate random costume id and hair id
# Generate random accessories

import codecs
import io
import struct
from collections import OrderedDict

import h5py
import hsluv
import numpy as np
from PIL import Image, ImageDraw, ImageFont

DEBUG = False


def debug_print(*args):
    if DEBUG:
        print(*args)


# MAX is inclusive
SIGN_UINT1_MAX = 0x7f
SIGN_PAIRS = 0x80
SIGN_PAIRS_MAX = 0x8f
SIGN_LIST = 0x90
SIGN_LIST_MAX = 0x9f
SIGN_STR = 0xa0
SIGN_STR_MAX = 0xbf
SIGN_FALSE = 0xc2
SIGN_TRUE = 0xc3
SIGN_LIST_ALTER = 0xc4
SIGN_FIXED_SIZE_LIST = 0xc5
SIGN_FLOAT4 = 0xca
SIGN_UINT1_ALTER = 0xcc
SIGN_UINT2 = 0xcd
SIGN_UINT4 = 0xce
SIGN_LONG_STR = 0xd9
SIGN_LONG_LIST = 0xdc
SIGN_LONG_PAIRS = 0xde


def parse_token(data, idx0):
    idx = idx0
    if idx >= len(data):
        return None, 0

    if data[idx] <= SIGN_UINT1_MAX:
        token = data[idx]
        idx += 1

    elif SIGN_PAIRS <= data[idx] <= SIGN_PAIRS_MAX:
        token_len = data[idx] - SIGN_PAIRS
        token = OrderedDict()
        idx += 1
        for i in range(token_len):
            key, delta_idx = parse_token(data, idx)
            idx += delta_idx
            value, delta_idx = parse_token(data, idx)
            idx += delta_idx
            token[key] = value

    elif SIGN_LIST <= data[idx] <= SIGN_LIST_MAX:
        token_len = data[idx] - SIGN_LIST
        token = []
        idx += 1
        for i in range(token_len):
            value, delta_idx = parse_token(data, idx)
            idx += delta_idx
            token.append(value)

    elif SIGN_STR <= data[idx] <= SIGN_STR_MAX:
        token_len = data[idx] - SIGN_STR
        try:
            token = data[idx + 1:idx + token_len + 1].decode()
        except UnicodeDecodeError:
            debug_print('STR', idx, data[idx:idx + token_len + 1])
            token = data[idx + 1:idx + token_len + 1]
        idx += token_len + 1

    elif data[idx] == SIGN_FALSE:
        token = False
        idx += 1

    elif data[idx] == SIGN_TRUE:
        token = True
        idx += 1

    elif data[idx] == SIGN_LIST_ALTER:
        token_len = data[idx + 1]
        token = []
        idx += 2
        for i in range(token_len):
            value, delta_idx = parse_token(data, idx)
            idx += delta_idx
            token.append(value)
        token = ('LIST_ALTER', token)

    elif data[idx] == SIGN_FIXED_SIZE_LIST:
        token_len = struct.unpack('>H', data[idx + 1:idx + 3])[0]
        token = []
        idx += 3
        max_idx = idx + token_len
        while idx < max_idx:
            # There may be an additional 0
            if data[idx + 4] == 0:
                token.append(0)
                idx += 1
            idx += 4  # Size of data chunk
            value, delta_idx = parse_token(data, idx)
            idx += delta_idx
            token.append(value)
        token = ('FIXED_SIZE_LIST', token)

    elif data[idx] == SIGN_FLOAT4:
        token = struct.unpack('>f', data[idx + 1:idx + 5])[0]
        idx += 5

    elif data[idx] == SIGN_UINT1_ALTER:
        debug_print('UINT1', idx, data[idx], data[idx + 1])
        token = data[idx + 1]
        idx += 2

    elif data[idx] == SIGN_UINT2:
        token = struct.unpack('>H', data[idx + 1:idx + 3])[0]
        idx += 3

    elif data[idx] == SIGN_UINT4:
        token = struct.unpack('>I', data[idx + 1:idx + 5])[0]
        idx += 5

    elif data[idx] == SIGN_LONG_STR:
        token_len = data[idx + 1]
        try:
            token = data[idx + 2:idx + token_len + 2].decode()
        except UnicodeDecodeError:
            debug_print('LONG_STR', idx, data[idx:idx + token_len + 2])
            token = data[idx + 2:idx + token_len + 2]
        idx += token_len + 2

    elif data[idx] == SIGN_LONG_LIST:
        token_len = struct.unpack('>H', data[idx + 1:idx + 3])[0]
        token = []
        idx += 3
        for i in range(token_len):
            value, delta_idx = parse_token(data, idx)
            idx += delta_idx
            token.append(value)

    elif data[idx] == SIGN_LONG_PAIRS:
        token_len = struct.unpack('>H', data[idx + 1:idx + 3])[0]
        token = OrderedDict()
        idx += 3
        for i in range(token_len):
            key, delta_idx = parse_token(data, idx)
            idx += delta_idx
            value, delta_idx = parse_token(data, idx)
            idx += delta_idx
            token[key] = value

    else:
        debug_print('?', idx, data[idx])
        token = ('?', data[idx])
        idx += 1

    delta_idx = idx - idx0
    return token, delta_idx


def parse_token_list(data):
    tokens = []
    idx = 0
    while idx < len(data):
        token, delta_idx = parse_token(data, idx)
        idx += delta_idx
        tokens.append(token)
    return tokens


def dump_token_with_len(token):
    data = dump_token(token)
    data = struct.pack('<I', len(data)) + data
    return data


def dump_token(token):
    if type(token) == tuple:
        if len(token) == 2:
            if token[0] == '?':
                debug_print('?', token[1])
                data = bytes([token[1]])
            elif token[0] == 'LIST_ALTER':
                data = (bytes([SIGN_LIST_ALTER, len(token[1])]) + b''.join(
                    [dump_token(x) for x in token[1]]))
            elif token[0] == 'FIXED_SIZE_LIST':
                data_list = []
                for x in token[1]:
                    if x == 0:
                        data_list.append(b'\x00')
                    else:
                        data_list.append(dump_token_with_len(x))
                data = b''.join(data_list)
                data = (bytes([SIGN_FIXED_SIZE_LIST]) + struct.pack(
                    '>H', len(data)) + data)
            else:
                raise Exception('Unknown token <{}>: {}'.format(
                    type(token), token))
        else:
            raise Exception('Unknown token <{}>: {}'.format(
                type(token), token))

    elif type(token) == list:
        if len(token) < 16:
            data = (bytes([SIGN_LIST + len(token)]) + b''.join(
                [dump_token(x) for x in token]))
        else:
            data = (bytes([SIGN_LONG_LIST]) + struct.pack('>H', len(token)) +
                    b''.join([dump_token(x) for x in token]))

    elif type(token) == OrderedDict:
        if len(token) < 16:
            data = (bytes([SIGN_PAIRS + len(token)]) + b''.join(
                [dump_token(k) + dump_token(v) for k, v in token.items()]))
        else:
            data = (bytes([SIGN_LONG_PAIRS]) + struct.pack(
                '>H', len(token)) + b''.join(
                    [dump_token(k) + dump_token(v) for k, v in token.items()]))

    elif type(token) == str:
        data = token.encode()
        if len(data) < 32:
            data = bytes([SIGN_STR + len(data)]) + data
        else:
            data = bytes([SIGN_LONG_STR, len(data)]) + data

    elif type(token) == int:
        if token <= SIGN_UINT1_MAX:
            data = bytes([token])
        elif token < 2**8:
            data = bytes([SIGN_UINT1_ALTER]) + bytes([token])
        elif token < 2**16:
            data = bytes([SIGN_UINT2]) + struct.pack('>H', token)
        else:
            data = bytes([SIGN_UINT4]) + struct.pack('>I', token)

    elif type(token) == float:
        data = bytes([SIGN_FLOAT4]) + struct.pack('>f', token)

    elif type(token) == bool:
        data = bytes([SIGN_TRUE if token else SIGN_FALSE])

    else:
        raise Exception('Unknown token <{}>: {}'.format(type(token), token))

    return data


def read_png(data, idx0):
    idx = idx0

    # PNG magic number
    assert data[idx:idx + 8] == b'\x89\x50\x4e\x47\x0d\x0a\x1a\x0a'

    idx += 8
    while True:
        chunk_len = struct.unpack('>I', data[idx:idx + 4])[0]
        chunk_type = data[idx + 4:idx + 8].decode()
        idx += chunk_len + 12
        if chunk_type == 'IEND':
            break

    img = data[idx0:idx]
    delta_idx = idx - idx0
    return img, delta_idx


def read_card(filename):
    with open(filename, 'rb') as f:
        card_data = f.read()

    # img1: cover, 252x352
    idx = 0
    img1, delta_idx = read_png(card_data, idx)
    idx += delta_idx

    # img2: head, 240x320
    idx += 33  # \x64\x00\x00\x00 【KoiKatuChara】 0.0.0
    img2, delta_idx = read_png(card_data, idx)
    idx += delta_idx

    # unknown_data is usually \xb7\x00\x00\x00
    unknown_data = card_data[idx:idx + 4]
    idx += 4

    lstinfo_token, delta_idx = parse_token(card_data, idx)
    idx += delta_idx

    has_kkex = (lstinfo_token['lstInfo'][0]['name'] == 'KKEx')

    idx += 8  # Size of lists
    idx += 4  # Size of face
    face_token, delta_idx = parse_token(card_data, idx)
    idx += delta_idx
    idx += 4  # Size of body
    body_token, delta_idx = parse_token(card_data, idx)
    idx += delta_idx
    idx += 4  # Size of hair
    hair_token, delta_idx = parse_token(card_data, idx)
    idx += delta_idx

    coordinate_token, delta_idx = parse_token(card_data, idx)
    idx += delta_idx
    parameter_token, delta_idx = parse_token(card_data, idx)
    idx += delta_idx
    status_token, delta_idx = parse_token(card_data, idx)
    idx += delta_idx

    if has_kkex:
        kkex_data = card_data[idx:]

    card = {
        'img1': img1,
        'img2': img2,
        'unknown_data': unknown_data,
        'lstInfo': lstinfo_token,
        'face': face_token,
        'body': body_token,
        'hair': hair_token,
        'coordinate': coordinate_token,
        'parameter': parameter_token,
        'status': status_token,
    }

    if has_kkex:
        card['KKEx'] = kkex_data

    return card


def write_card(filename, card):
    has_kkex = ('KKEx' in card)

    face_data = dump_token_with_len(card['face'])
    body_data = dump_token_with_len(card['body'])
    hair_data = dump_token_with_len(card['hair'])
    coordinate_data = dump_token(card['coordinate'])
    parameter_data = dump_token(card['parameter'])
    status_data = dump_token(card['status'])

    if has_kkex:
        # KKEx is not modified
        lst_idx = {
            'KKEx': 0,
            'Custom': 1,
            'Coordinate': 2,
            'Parameter': 3,
            'Status': 4,
        }
    else:
        lst_idx = {
            'Custom': 0,
            'Coordinate': 1,
            'Parameter': 2,
            'Status': 3,
        }

    idx = 0
    token = card['lstInfo']['lstInfo']
    token[lst_idx['Custom']]['pos'] = idx
    token[lst_idx['Custom']]['size'] = (
        len(face_data) + len(body_data) + len(hair_data))
    idx += len(face_data) + len(body_data) + len(hair_data)
    token[lst_idx['Coordinate']]['pos'] = idx
    token[lst_idx['Coordinate']]['size'] = len(coordinate_data)
    idx += len(coordinate_data)
    token[lst_idx['Parameter']]['pos'] = idx
    token[lst_idx['Parameter']]['size'] = len(parameter_data)
    idx += len(parameter_data)
    token[lst_idx['Status']]['pos'] = idx
    token[lst_idx['Status']]['size'] = len(status_data)
    idx += len(status_data)
    lstinfo_data = dump_token(card['lstInfo'])

    data_len = (len(face_data) + len(body_data) + len(hair_data) +
                len(coordinate_data) + len(parameter_data) + len(status_data))

    if has_kkex:
        data_len += len(card['KKEx'])

    with open(filename, 'wb') as f:
        f.write(card['img1'])

        f.write(b''.join([
            b'\x64\x00\x00\x00',
            b'\x12',
            '【KoiKatuChara】'.encode(),
            b'\x05',
            '0.0.0'.encode(),
            struct.pack('<I', len(card['img2'])),
        ]))
        f.write(card['img2'])

        f.write(card['unknown_data'])

        f.write(lstinfo_data)

        f.write(struct.pack('<Q', data_len))
        f.write(face_data)
        f.write(body_data)
        f.write(hair_data)

        f.write(coordinate_data)
        f.write(parameter_data)
        f.write(status_data)

        if has_kkex:
            f.write(card['KKEx'])


def generate_img_text(width, height, bg_color, text, text_color):
    font_name = 'simhei.ttf'
    font_size = 120

    img = Image.new('RGB', (width, height), bg_color)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_name, font_size)
    draw.text(((width - font_size) // 2, (height - font_size) // 2), text,
              text_color, font)

    bytes_io = io.BytesIO()
    img.save(bytes_io, format='PNG', optimize=True, compress_level=9)
    img_data = bytes_io.getvalue()

    return img_data


def generate_img12(name):
    hue = np.random.random() * 360
    bg_color = hsluv.hsluv_to_rgb([
        hue,
        50 + np.random.random() * 50,
        50 + np.random.random() * 50,
    ])
    bg_color = tuple(int(x * 256) for x in bg_color)
    text_color = hsluv.hsluv_to_rgb([
        (hue + 120 + np.random.random() * 120) % 360,
        50 + np.random.random() * 50,
        np.random.random() * 100,
    ])
    text_color = tuple(int(x * 256) for x in text_color)
    img1 = generate_img_text(252, 352, bg_color, name, text_color)
    img2 = generate_img_text(240, 320, bg_color, name, text_color)
    return img1, img2


def read_extern_img(filename):
    with open(filename, 'rb') as f:
        img_data = f.read()
    return img_data


def read_mean_cov(filename):
    with h5py.File(filename, 'r') as f:
        mean = np.array(f['mean'], dtype=float)
        cov = np.array(f['cov'], dtype=float)
    return mean, cov


def generate_params(mean, cov):
    params = np.random.multivariate_normal(mean, cov)
    params = np.clip(params, 0, 1)
    params = params.tolist()
    return params


def read_body_params_model():
    return read_mean_cov('data/body_params.hdf5')


def generate_body_params(body_params_model):
    return generate_params(*body_params_model)


# Eyebrow color and underhair color will be set with hair color
# Hair length, position, acsColor are not set yet
body_config = [
    'face.shapeValueFace',
    'face.detailPower',
    'face.cheekGlossPower',
    'face.pupil.0.baseColor.0:3',
    'face.pupil.0.subColor.0:3',
    'face.pupil.0.gradBlend',
    'face.pupil.0.gradOffsetY',
    'face.pupil.0.gradScale',
    'face.hlUpColor',
    'face.hlDownColor',
    'face.whiteBaseColor.0:3',
    'face.whiteSubColor.0:3',
    'face.pupilWidth',
    'face.pupilHeight',
    'face.pupilX',
    'face.pupilY',
    'face.eyelineUpWeight',
    'face.eyelineColor.0:3',
    'face.moleColor',
    'face.moleLayout',
    'face.lipLineColor',
    'face.lipGlossPower',
    'face.baseMakeup.eyeshadowColor',
    'face.baseMakeup.cheekColor',
    'face.baseMakeup.lipColor',
    'face.baseMakeup.paintColor.0',
    'face.baseMakeup.paintColor.1',
    'face.baseMakeup.paintLayout.0',
    'face.baseMakeup.paintLayout.1',
    'body.shapeValueBody',
    'body.bustSoftness',
    'body.bustWeight',
    'body.detailPower',
    'body.skinMainColor.0:3',
    'body.skinSubColor.0:3',
    'body.skinGlossPower',
    'body.paintColor.0',
    'body.paintColor.1',
    'body.paintLayout.0',
    'body.paintLayout.1',
    'body.sunburnColor',
    'body.nipColor.0:3',
    'body.nipGlossPower',
    'body.areolaSize',
    'body.nailColor.0:3',
    'body.nailGlossPower',
    'hair.parts.0.baseColor.0:3',
    'hair.parts.0.startColor.0:3',
    'hair.parts.0.endColor.0:3',
    'hair.parts.0.outlineColor.0:3',
    'hair.parts.1.baseColor.0:3',
    'hair.parts.1.startColor.0:3',
    'hair.parts.1.endColor.0:3',
    'hair.parts.1.outlineColor.0:3',
    'hair.parts.2.baseColor.0:3',
    'hair.parts.2.startColor.0:3',
    'hair.parts.2.endColor.0:3',
    'hair.parts.2.outlineColor.0:3',
    'hair.parts.3.baseColor.0:3',
    'hair.parts.3.startColor.0:3',
    'hair.parts.3.endColor.0:3',
    'hair.parts.3.outlineColor.0:3',
]


def get_child(card, path):
    child = card
    keys = path.split('.')
    for key in keys:
        if key in '0123456789':
            key = int(key)
        elif ':' in key:
            start, end = key.split(':')
            key = slice(int(start), int(end))
        child = child[key]
    return child


def set_child(card, path, value):
    child = card
    keys = path.split('.')
    count = 0
    for key in keys:
        if key in '0123456789':
            key = int(key)
        elif ':' in key:
            start, end = key.split(':')
            key = slice(int(start), int(end))
        count += 1
        if count == len(keys):
            child[key] = value
        else:
            child = child[key]


def parse_body_params(card):
    out = []
    for path in body_config:
        param = get_child(card, path)
        if type(param) == list:
            out += param
        else:
            out.append(param)
    out = np.array(out, dtype=float)
    return out


def dump_body_params(card,
                     body_params,
                     copy_pupil=False,
                     copy_hair_color=False):
    idx = 0
    for path in body_config:
        param = get_child(card, path)
        if type(param) == list:
            delta_idx = len(param)
            set_child(card, path, body_params[idx:idx + delta_idx])
            idx += delta_idx
        else:
            set_child(card, path, float(body_params[idx]))
            idx += 1
    assert idx == len(body_params)
    if copy_pupil:
        card['face']['pupil'][1] = card['face']['pupil'][0]
    if copy_hair_color:
        hair_color = card['hair']['parts'][0]['baseColor']
        card['face']['eyebrowColor'] = hair_color
        card['body']['underhairColor'] = hair_color


def parse_hair_ids(card):
    return [hair['id'] for hair in card['hair']['parts']]


def dump_hair_ids(card, hair_ids):
    for hair, hair_id in zip(card['hair']['parts'], hair_ids):
        hair['id'] = hair_id


def parse_costume_ids(card):
    return [[part['id'] for part in coordinate[1][0]['parts']]
            for coordinate in card['coordinate']]


def dump_costume_ids(card, costume_ids):
    for coordinate, part_ids in zip(card['coordinate'], costume_ids):
        for part, part_id in zip(coordinate[1][0]['parts'], part_ids):
            part['id'] = part_id


def read_costume_colors_model():
    return read_mean_cov('data/costume_colors.hdf5')


def generate_costume_colors(costume_colors_model):
    return [generate_params(*costume_colors_model) for i in range(7)]


# TODO: pattern
def parse_costume_colors(card):
    return [
        sum([
            color['baseColor'][:3] for part in coordinate[1][0]['parts']
            for color in part['colorInfo']
        ], []) for coordinate in card['coordinate']
    ]


def dump_costume_colors(card, costume_colors):
    for coordinate, part_colors in zip(card['coordinate'], costume_colors):
        for part_idx, part in enumerate(coordinate[1][0]['parts']):
            for color_idx, color in enumerate(part['colorInfo']):
                idx = (part_idx * 4 + color_idx) * 3
                color['baseColor'][:3] = part_colors[idx:idx + 3]


def read_name_data():
    with codecs.open('data/last_name.txt', 'r', 'utf-8') as f:
        last_names = [line.strip() for line in f]
    with codecs.open('data/male_name.txt', 'r', 'utf-8') as f:
        male_names = [line.strip() for line in f]
    with codecs.open('data/female_name.txt', 'r', 'utf-8') as f:
        female_names = [line.strip() for line in f]
    return last_names, male_names, female_names


GENDER_MALE = 0
GENDER_FEMALE = 1


def generate_name(last_names, male_names, female_names, gender=GENDER_FEMALE):
    # Convert np.str to builtin str
    last_name = str(np.random.choice(last_names))

    if gender == GENDER_MALE:
        first_name = str(np.random.choice(male_names))
    elif gender == GENDER_FEMALE:
        first_name = str(np.random.choice(female_names))
    else:
        raise Exception('Unknown gender: {}'.format(gender))
    hiragana, first_name = first_name.split()

    if len(hiragana) == 1:
        nickname = hiragana
    else:
        choice = np.random.randint(5)
        if choice == 0:
            nickname = hiragana[0]
        elif choice == 1:
            nickname = hiragana[-1]
        elif choice == 2:
            nickname = hiragana[:2]
        elif choice == 3:
            nickname = hiragana[-2:]
        else:  # choice == 4
            nickname = hiragana

    suffix = str(np.random.choice(['ちゃん', 'たん', 'りん', 'じん']))
    nickname += suffix

    return last_name, first_name, nickname


def parse_name(card):
    return (card['parameter']['lastname'], card['parameter']['firstname'],
            card['parameter']['nickname'])


def dump_name(card, last_name, first_name, nickname):
    card['parameter']['lastname'] = last_name
    card['parameter']['firstname'] = first_name
    card['parameter']['nickname'] = nickname
