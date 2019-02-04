# Test read + write is identity

from pykoikatu import *

in_filename = 'in.png'
out_filename = 'out.png'


def test_card(in_filename, out_filename):
    card = read_card(in_filename)
    dump_body_params(card, parse_body_params(card))
    dump_hair_ids(card, parse_hair_ids(card))
    dump_costume_ids(card, parse_costume_ids(card))
    dump_costume_colors(card, parse_costume_colors(card))
    dump_name(card, *parse_name(card))
    write_card(out_filename, card)

    with open(in_filename, 'rb') as f:
        card_data_in = f.read()
    with open(out_filename, 'rb') as f:
        card_data_out = f.read()
    # card_data_in may have additional data at the end, such as eof and bepis
    assert card_data_in.startswith(card_data_out)


if __name__ == '__main__':
    test_card(in_filename, out_filename)
