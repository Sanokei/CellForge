Design: parameter-free cluster cl_qstate. y = (1 - p_bos) * (q + p_quote - 2*q*p_quote), where p_quote = prev[9], p_bos = prev[1]
(slice/mul/add/scale/bias ops; exact on 0/1 inputs). Toggle when quote, reset to 0 on bos, else keep q.
Params: 0. Cells: 0 (ops only). Cases: all passed (see component-test).
Contract concerns: none.
