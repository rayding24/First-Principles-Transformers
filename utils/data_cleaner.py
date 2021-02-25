
import re

def basic_cleaning(input_path='data/haiku.txt', output_path='data/haiku_cleaned.txt', threshold=50):
    '''
    Ignore lines that exceeds threshold length for poem,
    and lines starting with non alphabet 
    '''
    with open(input_path, 'r') as fsrc:
        with open(output_path, 'w') as fdest:
            for line in fsrc:
                if len(line)>50 or not line[0].isalpha():
                    continue
                else:
                    fdest.write(line)

def charset_cleaning(input_path='data/haiku_cleaned.txt', output_path='data/haiku_cleaned1.txt', charset='[#$@%&*():<>[\]{}~=1234567890]'):
    '''
    Ignore lines that contain special chars from charset
    '''
    regex = re.compile(charset)
    with open(input_path, 'r') as fsrc:
        with open(output_path, 'w') as fdest:
            for line in fsrc:
                if regex.search(line) == None:
                    fdest.write(line)



if __name__ == '__main__':
    basic_cleaning()
    charset_cleaning()