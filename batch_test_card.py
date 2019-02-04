import os

from test_card import test_card

in_dir = 'tmp'
err_dir = 'tmp2'
out_filename = 'out.png'

if __name__ == '__main__':
    for root, dirs, files in os.walk(in_dir):
        for file in sorted(files):
            in_filename = os.path.join(root, file)
            print(in_filename)
            try:
                test_card(in_filename, out_filename)
            except Exception as e:
                print('Error', e)
                os.rename(in_filename, in_filename.replace(in_dir, err_dir))
