PROBABLY_THE_SAME_MAX_CHANGES = 5

def are_those_plates_probably_the_same(plate_a, plate_b):
	return restricted_edit_distance(plate_a, plate_b) < PROBABLY_THE_SAME_MAX_CHANGES

def restricted_edit_distance(a, b):
	"""Damerau-Levenshtein with restricted edit distance/optimal string alignment distance"""
	if a == b:
		return 0

	d = {}
	for i in range(-1, len(a)+1):
		d[(i, -1)] = i+1
	for j in range(-1, len(b)+1):
		d[(-1, j)] = j+1

	for i in range(len(a)):
		for j in range(len(b)):
			if a[i] == b[j]:
				cost = 0
			else:
				cost = 1
			d[(i, j)] = min(
				d[(i-1, j)] + 1, # deletion
				d[(i, j-1)] + 1, # insertion
				d[(i-1, j-1)] + cost # substitution
			)
			if i > 1 and j > 1 and a[i] == b[j-1] and a[i-1] == b[j]:
				d[(i, j)] = min(d[(i, j)], d[i-2, j-2] + cost) # transposition

	return d[len(a)-1, len(b)-1]
