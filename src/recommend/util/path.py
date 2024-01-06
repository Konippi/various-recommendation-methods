import os


class Path:
    def __init__(self):
        pass

    def get_root(self) -> str:
        """
        Get the root directory of the project.

        Parameters
        ----------
            None

        Returns
        -------
            root_dir: Path
                The root directory of the project
        """
        _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname((os.path.abspath(__file__))))))
        return _root + os.path.sep
