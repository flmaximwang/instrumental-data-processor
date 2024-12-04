import os

def get_name_from_path(file_path, extension=True):
    basename = os.path.basename(file_path)
    if extension:
        return ".".join(basename.split('.')[:-1])
    else:
        return basename

def get_name_from_basename(basename, extension=True):
    if extension:
        return ".".join(basename.split('.')[:-1])
    else:
        return basename