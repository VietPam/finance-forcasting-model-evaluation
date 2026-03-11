import os

root_dir="."
output_file="merged_python_files.txt"

with open(output_file,"w",encoding="utf-8") as outfile:
    for root,dirs,files in os.walk(root_dir):
        for file in files:
            if file.endswith(".py") and file!=output_file:
                file_path=os.path.join(root,file)
                with open(file_path,"r",encoding="utf-8") as infile:
                    content=infile.read()
                    content=content.replace("\n"," ").replace("\r"," ")
                    outfile.write("FILE:"+file_path+" "+content+"\n")