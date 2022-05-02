

T4sa = {
    "layer":4,
    "hidden_size":1024,
    "dim":1024,
    "dropout_r":0.1,
    "multi_head":8,
    "ff_size":2048,
    "word_embed_size":300,
    "image_hw":256,
    "patch_hw":32,
    #"lang_size":31,
    "num_classes":3,
    "img_len":65,
    "text_len":31
}


Mosei = {
    "layer":4,
    "hidden_size":1024,
    "dim":1024,
    "dropout_r":0.1,
    "multi_head":8,
    "ff_size":2048,
    "word_embed_size":300,
    "num_classes":2,
    "audio_len":61,
    "text_len":61
}

Mosei_early = {
    "layer":4,
    "hidden_size":1024,
    "dim":1024,
    "dropout_r":0.1,
    "multi_head":8,
    "ff_size":2048,
    "word_embed_size":300,
    "num_classes":2,
    "audio_len":60,
    "text_len":61
}

SNLI = {
    "layer":4,
    "hidden_size":1024,
    "dim":1024,
    "dropout_r":0.1,
    "multi_head":8,
    "ff_size":2048,
    "word_embed_size":300,
    "image_hw":256,
    "patch_hw":32,
    "num_classes":3,
    "text_len":51

}





class Config():
    def __init__(self,args_dict):
        self.args_dict = args_dict 
        self.add_args(self.args_dict)
        self.model_param_select()
    
    def add_args(self, args_dict):
        for arg in args_dict.keys():
            setattr(self, arg, args_dict[arg])


    def model_param_select(self):
        if self.dataset =="T4sa":
            self.add_args(T4sa)
        
        elif self.dataset == "Mosei" and self.model != "Early":
            self.add_args(Mosei)
        elif self.dataset == "Mosei" and self.model == "Early":
            self.add_args(Mosei_early)

        elif self.dataset=="SNLI":
            self.add_args(SNLI)
        # if self.dataset =="T4sa" and self.model=="Early":
        #     self.add_args(T4sa_early)
