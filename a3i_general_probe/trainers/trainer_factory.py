
class TrainerRegister: 
    __data = {}
    @classmethod
    def register(cls,cls_name=None):
        def decorator(cls_obj):
            cls.__data[cls_name]=cls_obj
            return cls_obj
        return decorator
    @classmethod
    def get_trainer(cls,key):
        return cls.__data[key]
    @classmethod
    def num_trainers(cls):
        return len(cls.__data)
    @classmethod
    def get_trainers(cls):
        return cls.__data.keys()
def load_trainer(conf): 
    return TrainerRegister.get_trainer(conf['train_mode'])