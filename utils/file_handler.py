import shutil
import os

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



if __name__ == '__main__':
    merge_txt_files()


