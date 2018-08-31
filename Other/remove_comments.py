import re
import os
import codecs

pattern = r'[^a-zA-Z0-9 !@#$%^&*()-_=+{}:;<>,.?|\[\]\'\"\/\\]'
target_path = "C:/Users/Leon/Desktop"
extension = ".cs"
coding_format = "utf8"

for root, dirs, files in os.walk(target_path):
    for entry in files:
        if entry.endswith(extension):
            with open(os.path.join(root, entry), "r+", encoding = coding_format) as target_file:
                lines = target_file.read().splitlines()
                target_file.seek(0)

                for line in lines:
                    # Remove the line in which we find any symbols not in the pattern.
                    if not re.search(pattern, line):
                        target_file.write(line)
                        target_file.write("\n")

                target_file.truncate()