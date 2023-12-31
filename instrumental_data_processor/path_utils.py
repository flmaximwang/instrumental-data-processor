import os

def get_name_from_path(file_path, extension=True):
    basename = os.path.basename(file_path, extension)
    return ".".join(basename.split('.')[:-1])

def get_name_from_basename(basename, extension=True):
    return ".".join(basename.split('.')[:-1])