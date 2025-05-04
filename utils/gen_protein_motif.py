
motif_list = []
with open('utils/protein-motif.txt', 'r') as f:
    for line in f.readlines():
        motif_list.append(line.strip())


def collect_protein_motif_idx(seq):
    idx_list = []
    for idx, motif in enumerate(motif_list):
        if motif in seq:
            idx_list.append(idx)
    return idx_list


seq = 'MAKEDTLEFPGVVKELLPNATFRVELDNGHELIAVMAGKMRKNRIRVLAGDKVQVEMTPYDLSKGRINYRFK'
ids = collect_protein_motif_idx(seq)
print(ids)