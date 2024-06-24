from typing import (
    Sequence,
    List,
    Union
)
from abc import ABC, abstractmethod

import os


class AbstractFileSystem(ABC):

    base_path: str

    @abstractmethod
    def list_all(cls, exclude: Sequence[str], namespace: str) -> List[str]:

        """
            Returns all the files available
        """

        ...


class LocalFileSystem(AbstractFileSystem):

    def __init__(self, base_path: str = './images'):

        self.base_path = base_path
        self.persist_path()

    def persist_path(self, subpath: str = ''):

        """
            Makes sure the base path is always created
        """

        path = os.path.join(self.base_path, subpath)

        print(path, self.base_path, subpath)

        os.makedirs(path, exist_ok=True)

        return path

    def find_path(self, object_key) -> Union[str, None]:

        """
            Find the exact path where the object is located
        """

        paths = [self.base_path]

        while paths:

            current_path = paths.pop(0)

            if os.path.isfile(current_path):
                continue

            subdirs = os.listdir(current_path)

            if object_key in subdirs:

                return current_path

            paths += [os.path.join(current_path, subdir) for subdir in subdirs]

        return None

    def write_file(
        self,
        object_key: str,
        content: bytes,
        namespace: str = None
    ) -> None:

        if namespace is not None:
            path = os.path.join(self.base_path, namespace)

            os.makedirs(path, exist_ok=True)

        else:

            path = self.base_path

        filepath = os.path.join(path, object_key)

        with open(filepath, "wb") as f:

            f.write(content)

    def read_file(
        self,
        object_key,
        namespace: str = None
    ) -> bytes:

        if namespace is not None:
            path = os.path.join(self.base_path, namespace)
        else:
            path = self.find_path(object_key)

        if path is None:
            return None

        filepath = os.path.join(path, object_key)

        if not os.path.exists(filepath):
            return None

        return open(filepath, "rb").read()

    def delete_file(self, object_key) -> None:

        path = self.find_path(object_key)

        if path is None:
            return

        filepath = os.path.join(path, object_key)

        os.remove(filepath)

    def list_all(
        self,
        exclude: set = None,
        namespace: str = None
    ) -> List[str]:

        files = []

        if namespace is not None:
            paths = [os.path.join(self.base_path, namespace)]
        else:
            paths = [self.base_path]

        exclude = {
            os.path.join(self.base_path, path)
            for path in exclude
        } if exclude is not None else set()

        while paths:

            current_path = paths.pop(0)

            # skip the paths that we don't want to include
            if current_path in exclude:
                continue

            if not os.path.exists(current_path):
                break

            subpaths = [
                os.path.join(current_path, key)
                for key in os.listdir(current_path)
            ]

            files += [
                subpath
                for subpath in subpaths
                if os.path.isfile(subpath)
            ]

            paths += [
                subpath
                for subpath in subpaths
                if os.path.isdir(subpath)
            ]

        return files

    def read_all(self, namespace: str = None) -> List[bytes]:

        return [
            open(filepath, 'rb').read()
            for filepath in self.list_all(namespace)
        ]
