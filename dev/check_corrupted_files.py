import os
import numpy as np

"""
File ~/bam_ws/venv/lib/python3.12/site-packages/numpy/lib/npyio.py:256, in NpzFile.__getitem__(self, key)
    254 if magic == format.MAGIC_PREFIX:
...
   1005 # Check the CRC if we're at the end of the file
   1006 if self._eof and self._running_crc != self._expected_crc:
-> 1007     raise BadZipFile("Bad CRC-32 for file %r" % self.name)

BadZipFile: Bad CRC-32 for file 'offsets.npy'

Perhaps caused by juypter closing while these large files are being read/written
"""
grasp_label_dir = '/home/bam/graspnetAPI/grasp_label'
for fname in os.listdir(grasp_label_dir):
    if fname.endswith('.npz'):
        fpath = os.path.join(grasp_label_dir, fname)
        try:
            np.load(fpath)
        except Exception as e:
            print(f"Corrupted: {fname} - {e}")