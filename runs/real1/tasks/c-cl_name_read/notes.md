# cl_name_read notes
Design: exact ops only, no cells, one param (`cl_name_read.endvec`, 36 elements = word-end profile from contract data).
- nm[0:216] -> [6,36] -> onehot_argmax per row = name letters; nm[217:224] -> onehot_argmax = L; nm[216] = cap.
- D = matmul(B, ctx^T) [6,8]: letter-match dot products. m_k = prod_j D[j,8-k+j] * boundary(row 7-k), k=1..6
  (boundary = 1 - letter_mask - apostrophe, so zero rows and bos/space/punct/quote count as boundaries).
- Gates: notcap = step(0.5-cap); L>k = sum Lo[k+1:7]; L==k = Lo[k]. L<2 excluded by only using Lo[2:7] / L==k for k>=2.
- Outputs: word-start (ctx row 7 space or quote) 1.2*name[0]; continue g_k*name[k] for k=1..5 when L>k; end at k==L (k=2..6) via endvec.
- Constants g_k, 1.2 are scale-op args typed from the contract data.
Params: 1 unique param (36 elements). Cells: 0 raw cells (192 op nodes, all in one cluster).
Component test: 15/15 passed.
Concerns: none known. Assumes nm buffer letters and L slots as in contract layout (row-major 6x36, L one-hot at 217..223); argmax makes it robust to soft values.
