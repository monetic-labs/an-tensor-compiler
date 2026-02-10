# Philosophy

## What This Is

A compiler that transforms symbolic logic rules into differentiable tensor
equations. You write rules in a Prolog-like DSL, and the compiler produces
GPU-accelerated functions with full gradient flow. The rules are learnable —
thresholds, projections, similarities, and graph structures all train via
standard backpropagation.

This is not a framework. It's a tool. Fork it, bend it, see what happens.

## Intellectual Lineage

### Pedro Domingos & Neuro-Symbolic Unification

The core insight — that symbolic rules can be compiled into differentiable
tensor operations without losing either the interpretability of logic or the
learnability of neural networks — draws directly from Pedro Domingos' work.
His research on unifying logical and statistical AI, particularly the
demonstration that gradient descent operates in kernel space, informed our
approach to making symbolic reasoning fully differentiable.

Key paper: "Every Model Learned by Gradient Descent Is Approximately a
Kernel Machine" (arXiv:2012.00152)

### Lotfi Zadeh & Fuzzy Logic

The differentiable operators at the foundation of this compiler — fuzzy AND,
OR, NOT, IMPLIES — extend Zadeh's continuous-valued logic (1965) into the
tensor domain. Where classical logic gives you {0, 1}, fuzzy logic gives you
[0, 1], and crucially, gradients can flow through [0, 1].

### Tony Plate & Holographic Reduced Representations

The HRR module implements Plate's circular convolution binding (1995) for
distributed representations. The key property: every piece of a hologram
contains information about the whole. This isn't metaphor — it's the
mathematical property that makes holographic memory work for compositional
structures.

## Biomimicry as Structural Principle

The patterns in this codebase — organisms, bounded contexts, federation,
holographic memory, gradient isolation — are not naming conventions borrowed
from biology for aesthetic reasons. They reflect structural principles
observed in natural systems that solve the same problems we face in
distributed learning:

**Composition over monoliths.** Biological systems build complex behavior
from simple, composable units. A cell doesn't contain a blueprint for the
whole organism — it contains local rules that produce global behavior through
composition. The compiler works the same way: simple predicates compose into
complex decisions through fuzzy logic operators.

**Locality with global coherence.** Neurons maintain local state but
participate in global computation through connection patterns. Namespaces
provide gradient isolation (local state) while federation enables cross-domain
learning (global coherence). Neither is sacrificed for the other.

**Distributed representation.** Biological memory isn't stored in a single
location — it's distributed across neural populations. Holographic reduced
representations achieve the same property mathematically: binding creates
distributed encodings where information about the whole is present in every
part.

**Robustness through redundancy.** Natural systems don't have single points
of failure. CRDTs enable multi-device tensor synchronization with the same
principle — eventual consistency without coordination, convergence without
consensus.

We study natural systems not to mimic biology with the wrong paradigms, but
to find paradigms that actually work. When biological systems and
computational systems face the same structural problem, the biological
solution — refined by billions of years of selection pressure — is worth
understanding deeply before inventing something new.

## How to Engage

Pull requests are welcome. We review, we decide. If your change reveals
something we couldn't see, it gets in.

Fork freely. The compiler is designed with clear extension points — new
predicate types, new similarity methods, new aggregation strategies. The
best outcome is someone in a completely different domain discovering that
the same differentiable logic architecture solves their problem in a way
we never anticipated.

If you're an academic: cite as you see fit, build on it, publish your
findings. That's the whole point.
