import shelve

class Indexer:
    def __init__(self, source_file = "mapping.sv"):
        self.isOpen = False
        self.open(source_file)

    def open(self, source_file):
        if self.isOpen is True:
            raise IOError("already opened %s" % self.filename)
        self.dic = self.__initial_dictionary()
        self.__sh = shelve.open(source_file)
        self.__read(self.__sh)
        self.filename = source_file
        self.isOpen = True

    def __initial_dictionary(self, predict_target="chunk", other_label="O"):
        mode_candidates = ["char", "POS", "chunk"]
        output = {mode:{"UNK":0} for mode in mode_candidates if mode is not predict_target}
        output[predict_target] = {other_label:0, "UNK":1}
        return output

    def __read(self, source_sh, overwrite=True):
        for key in source_sh.keys():
            if overwrite:
                self.dic[key] = source_sh[key]
            else:
                if key not in self.dic:
                    self.dic[key] = source_sh[key]

    def __del__(self):
        self.close()

    def close(self):
        if self.isOpen is not True:
            return
        self.update()
        self.__sh.close()
        self.isOpen = False

    def update(self):
        for k,v in self.dic.items():
            self.__sh[k] = v
        self.__sh.sync()

    def ConvertToID(self, key, mode, useUNK=False):
        if key not in self.dic[mode]:
            value = len(self.dic[mode])
            self.dic[mode][key] = value
        else:
            value = self.dic[mode][key]
        return value

