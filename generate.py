# Generate a random character

from pykoikatu import *

in_filename = 'in.png'
out_filename = 'out_{}.png'

if __name__ == '__main__':
    card = read_card(in_filename)
    for count in range(10):
        print(count)
        dump_body_params(
            card,
            generate_body_params(),
            copy_pupil=True,
            copy_hair_color=True)
        dump_hair_ids(card, generate_hair_ids())
        dump_costume_ids(card, generate_costume_ids())
        dump_costume_colors(card, generate_costume_colors())

        last_name, first_name, nickname = generate_name()
        dump_name(card, last_name, first_name, nickname)

        card['img1'], card['img2'] = generate_img12(last_name[0])

        write_card(out_filename.format(last_name + first_name), card)
