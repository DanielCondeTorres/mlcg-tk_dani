import torch
from typing import Optional

@torch.jit.script
def compute_raw_angles(
    pos: torch.Tensor,
    mapping: torch.Tensor,
    cell_shifts: Optional[torch.Tensor] = None,
):
    r"""Compute the raw value of the angle between the positions in :obj:`pos` following the :obj:`mapping` assuming that the mapping indices follow::

       j--k
      /
     i

    .. math::

        \cos{\theta_{ijk}} &= \frac{\mathbf{r}_{ji} \mathbf{r}_{jk}}{r_{ji} r_{jk}}  \\
        r_{ji}&= ||\mathbf{r}_i - \mathbf{r}_j||_{2} \\
        r_{jk}&= ||\mathbf{r}_k - \mathbf{r}_j||_{2}

    In the case of periodic boundary conditions, :obj:`cell_shifts` must be
    provided so that :math:`\mathbf{r}_j` can be outside of the original unit
    cell.
    """
    assert mapping.dim() == 2
    assert mapping.shape[0] == 3

    dr1 = pos[mapping[0]] - pos[mapping[1]]
    dr2 = pos[mapping[2]] - pos[mapping[1]]
    n = torch.cross(dr1,dr2)
    n = n.norm(p=2, dim=1)
    d = (dr1 * dr2).sum(dim=1)
    theta = torch.atan2(n,d) 
    return theta
