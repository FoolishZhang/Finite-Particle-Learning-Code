"""Selected saved-arithmetic functions; no producer entry points."""

from __future__ import annotations

import common

from flint import arb,arb_mat,ctx

def zero_matrix(m):return [[arb(0) for _ in range(m)] for _ in range(m)]

def as_arb(value):
    return value if isinstance(value,arb) else common.aq(value)

def matrix_records(matrix):return [[common.record(as_arb(value)) for value in row] for row in matrix]

def aggregate_records(rows):
    total={name:zero_matrix(8) for name in ('Hbb','Haa','Hperp')}
    for row in rows:
        for name in total:
            matrix=row['matrix_blocks'][name]
            if len(matrix)!=8 or any(len(values)!=8 for values in matrix):raise ValueError('full8x8 checkpoint blocks required')
            for i in range(8):
                for j in range(8):total[name][i][j]+=common.from_record(matrix[i][j])
    return {name:matrix_records(matrix) for name,matrix in total.items()}
