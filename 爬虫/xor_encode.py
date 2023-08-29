from pathlib import Path
from tqdm import tqdm
import xor_bytes
import os
import re

THREAD = 4
READ = 2**24

def xor_encrypt(block, key):
    # 对每个字节进行异或运算，并返回加密后的字节对象
    return bytes(b ^ key for b in block)

def encode(input_file,use = True):
    file_input = open(input_file,'rb')
    file_output = open(input_file+'.enc','wb')

    filesize = os.path.getsize(input_file)
    r = filesize % READ
    c = filesize // READ
    if r:
        t = tqdm(total = c, leave=True)
    else:
        t = tqdm(total = c + 1, leave=True)

    block = file_input.read(READ)
    while block:
        if use:
            xor_block = xor_bytes.convert(block,88,THREAD)
        else:
            xor_block = xor_encrypt(block,88)
        if len(xor_block) != len(block):
            raise Exception('Error')
        else:
            file_output.write(xor_block)
        block = file_input.read(READ)
        t.update(1)

    file_input.close()
    file_output.close()

if __name__ == '__main__':
    while True:
        path = input("请拖入要加密的文件:")
        res = re.findall(r"\"(.*?)\"", path)
        if len(res) == 0:
            res = [path]
        for path in res:
            if os.path.isdir(path):
                for root, dirs, files in os.walk(path):
                    for f in files:
                        file = Path(root).joinpath(f)
                        encode(str(file))
            else:
                encode(path)