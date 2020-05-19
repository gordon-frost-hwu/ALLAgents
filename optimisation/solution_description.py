#! /usr/bin/python


class SolutionDescription(object):
    def __init__(self, num_genes, gene_bounds, gene_init_range,
                 gene_sigma, gene_mut_probability, mutation_type, atol):
        # assert all(isinstance(elem, list) and len(elem) == 2 for elem in gene_bounds),
        #               "gene_bounds must be a list of list"
        assert gene_bounds.shape == (num_genes, 2), \
            "gene_bounds must be numpy array of shape: ({0}, 2)".format(num_genes)
        assert gene_init_range.shape == (num_genes, 2), \
            "gene_init_range must be numpy array of shape: ({0}, 2)".format(num_genes)
        assert gene_sigma.shape == (num_genes, ), \
            "gene_sigma must be numpy array of shape: ({0}, )".format(num_genes)
        assert gene_mut_probability.shape == (num_genes, ), \
            "gene_mut_probability must be numpy array of shape: ({0}, )".format(num_genes)
        assert atol.shape == (num_genes, ), \
            "atol must be numpy array of shape: ({0}, )".format(num_genes)
        assert len(mutation_type) == num_genes, \
            "mutation_type must be list of size: {0}".format(num_genes)

        self.num_genes = num_genes
        self.gene_bounds = gene_bounds
        self.gene_init_range = gene_init_range
        # TODO - compute the signma from bounds?
        self.gene_sigma = gene_sigma
        self.gene_mutation_prob = gene_mut_probability
        self.gene_mutation_type = mutation_type
        self.atol = atol
