import io
import pickle
# Source is Stackoverflow https://stackoverflow.com/questions/2121874/python-pickling-after-changing-a-modules-directory

class RenameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        renamed_module = module
        if module == "question_answer_set":
            renamed_module = "utils.question_answer_set"

        return super(RenameUnpickler, self).find_class(renamed_module, name)


def renamed_load(file_obj):
    return RenameUnpickler(file_obj).load()


def renamed_loads(pickled_bytes):
    file_obj = io.BytesIO(pickled_bytes)
    return renamed_load(file_obj)
