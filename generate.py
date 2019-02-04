# Generate a random character

from pykoikatu import *

in_filename = 'in.png'
out_filename = 'out_{}.png'

if __name__ == '__main__':
    card = read_card(in_filename)
    body_params_model = read_body_params_model()
    costume_colors_model = read_costume_colors_model()
    last_names, male_names, female_names = read_name_data()
    for count in range(10):
        print(count)

        body_params = generate_body_params(body_params_model)
        dump_body_params(
            card, body_params, copy_pupil=True, copy_hair_color=True)

        costume_colors = generate_costume_colors(costume_colors_model)
        dump_costume_colors(card, costume_colors)

        last_name, first_name, nickname = generate_name(
            last_names, male_names, female_names)
        dump_name(card, last_name, first_name, nickname)

        card['img1'], card['img2'] = generate_img12(last_name[0])

        write_card(out_filename.format(last_name + first_name), card)
