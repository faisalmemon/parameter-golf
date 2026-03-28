Excellent baseline run. 276 steps, final val_bpb: 1.6951. This is your comparison number for every future experiment.

The most important finding — skip_weights collapse:

Skip	Connects	Step 0	Step 276
S0	L2→L3 (adjacent)	1.000	0.973
S1	L1→L4 (mid-range)	1.000	0.460
S2	L0→L5 (long-range)	1.000	0.257

The network is progressively suppressing the longer-range skips. The early encoder features (L0) barely reach the final decoder layer — S2 collapsed to 1/4 of its initial value. The adjacent skip (L2→L3) is nearly unchanged. This is a clear architectural signal: the U-Net skip connections are not symmetrically useful. The benefit comes almost entirely from the adjacent encoder→decoder skip, not the long-range ones.

q_gain — approaching dangerous territory:

L4:2.877, L5:3.031 at step 276 and still rising. These are decoder layers with skip connection inputs. Above ~3.5, attention entropy typically collapses (each position attends to 1–2 tokens only). Worth watching in the next run. If it exceeds 3.5, adding a q_gain clamp is warranted.

resid_mix L0 went positive (+0.138):
Every other layer subtracts from x0, but L0 (first encoder layer) is adding more of the initial embedding back. The first layer is learning to amplify token identity. Unusual but not harmful.

BPB curve still declining at step 276 — not converged. With more steps it would improve further, which makes the comparative signal slightly noisy. But it's consistent enough for ablations.

Suggested experiment order (each takes one 600s run):

MLP_MULT=3 — add one line to run_proxy.sh: export MLP_MULT=3. Widening MLP is the single most common change in the top records. Hypothesis: BPB drops ~0.05–0.10.

SwiGLU activation — replace relu² in MLP.forward:

Records show ~0.035 BPB improvement from this.

Remove long-range skips — given S2 collapsed to 0.257, test whether removing S1+S2 entirely and keeping only the adjacent skip (S0) hurts or helps. Hypothesis: negligible difference or slight improvement because the optimizer is already doing this anyway.

Want me to implement the SwiGLU change in MLP and adjust the run_proxy.sh to export MLP_MULT=3 so you can run both at once?
