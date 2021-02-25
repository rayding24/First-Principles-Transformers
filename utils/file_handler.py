import shutil
import os
import wget

def merge_txt_files(input_folder='data/forms/haiku', output_path='data/haiku.txt'):
    '''
    Merge all txt files in input_folder into one txt file
    by appending each txt into the output txt
    e.g. all haiku related txt into haiku.txt
    '''
    files = os.listdir(input_folder)
    files = [os.path.join(input_folder, file) for file in files]
    with open(output_path, 'wb') as fdest:
        for file in files:
            with open(file, 'rb') as fsrc:
                shutil.copyfileobj(fsrc, fdest)
                fdest.write(b'\n') #start new line for next file


def wget_file(output_path='data/haikuzao.txt', url='https://github.com/herval/creative_machines/blob/master/haikuzao/src/main/resources/haiku.txt'):
    '''Get file from web and save to output path'''
    # r = requests.get(url)

    # with open(output_path, 'wb') as f:
    #     f.write(r.content)
    wget.download(url, output_path)






if __name__ == '__main__':
    # merge_txt_files()
    wget_file()


