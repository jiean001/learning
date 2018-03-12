pre_config_file = "_pytorch_white_config.txt"
config_file = "pytorch_white_config.txt"

class Pre_Data:
    def __init__(self, pre_config_file, config_file):
        self.pre_config_file = pre_config_file
        self.config_file = config_file
        self.config_file_fp = open(self.config_file, "w+")
        self.base_len = len("\t")
        self.process()
        self.config_file_fp.close()

    def write_2_config_file(self, line):
        self.config_file_fp.write(line)

    def get_standard_letter(self, full_letter):
        index = full_letter.index("_")
        return full_letter[index+1] + ".jpg"

    def process(self):
        with open(self.pre_config_file,'r') as f:
            lines = f.readlines()
            # lines.sort(key=lambda x: len(x))
            lines.sort(key=lambda x: x.strip().count('.jpg'))
            for line in lines:
                for i in range(line.count("\t")):
                    letter = line.split("\t")
                    tmp_line = line.replace("\n", "").replace(letter[i] + "\t", "") + ": " + self.get_standard_letter(letter[i]) + " : " + letter[i] + "\n"
                    self.write_2_config_file(tmp_line.replace("\t", " "))
                '''
                if self.num < line.count("\t"):
                    self.num = line.count("\t")
                    self.print_word_len()
                '''


if __name__ == '__main__':
    Pre_Data(pre_config_file, config_file)