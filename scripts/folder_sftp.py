import paramiko
import os
# The following line is necessary to determine if a path on the SFTP server is a file.
from stat import S_ISREG

class FolderSFTPClient(paramiko.SFTPClient):
    def put_dir(self, source, target):
        """Uploads the contents of the source directory to the target path."""
        for item in os.listdir(source):
            if os.path.isfile(os.path.join(source, item)):
                self.put(os.path.join(source, item), '%s/%s' % (target, item))
            else:
                self.mkdir('%s/%s' % (target, item), ignore_existing=True)
                self.put_dir(os.path.join(source, item), '%s/%s' % (target, item))

    def get_dir(self, source, target):
        """Downloads the contents of the source directory from the SFTP server to the local target path."""
        if not os.path.exists(target):
            os.makedirs(target, exist_ok=True)

        for item in self.listdir(source):
            if self.isfile(os.path.join(source, item)):
                self.get(os.path.join(source, item), os.path.join(target, item))
            else:
                os.mkdir(os.path.join(target, item))
                self.get_dir(os.path.join(source, item), os.path.join(target, item))

    def isfile(self, path):
        """Determines if the path on the SFTP server is a file."""
        try:
            return S_ISREG(self.stat(path).st_mode)
        except IOError:
            # This might mean that the path doesn't exist, in which case it's not a file.
            return False

    def mkdir(self, path, mode=511, ignore_existing=False):
        """Augments mkdir by adding an option to not fail if the folder exists."""
        try:
            super(FolderSFTPClient, self).mkdir(path, mode)
        except IOError:
            if ignore_existing:
                pass
            else:
                raise



