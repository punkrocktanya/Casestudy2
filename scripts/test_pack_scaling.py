import unittest
import numpy as np
from scale_module_pack import envelope


class TopologyChecks(unittest.TestCase):
    def test_identical_cells_and_limiting_group(self):
        q=np.full((1,3,4,2),2.)
        modules,pack=envelope(q,np.ones_like(q))
        np.testing.assert_allclose(modules,4.)
        self.assertEqual(pack[0],4.)
        q[0,1,2,:]=1.
        modules,pack=envelope(q,np.ones_like(q))
        self.assertEqual(pack[0],2.)
        np.testing.assert_allclose(modules,[[4,2,4]])

    def test_soc_accessibility(self):
        q=np.full((1,1,2,2),2.)
        _,pack=envelope(q,np.full_like(q,.8))
        self.assertAlmostEqual(pack[0],3.2)


if __name__=='__main__':unittest.main()
