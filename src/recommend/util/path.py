import os


class Path:
    def __init__(self):
        pass

    def get_root_dir(self) -> str:
        """
        Get the root directory of the project.

        Parameters
        ----------
            None

        Returns
        -------
            root: str
                root directory of the project
        """
        __root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname((os.path.abspath(__file__))))))
        return __root + os.path.sep

    def get_local_path(self, path_from_root: str) -> str:
        """
        Get the path from the root directory of the project.

        Parameters
        ----------
            path_from_root: str
                path from the root directory of the project

        Returns
        -------
            local_path: str
                local path
        """
        return os.path.join(self.get_root_dir() + path_from_root)
