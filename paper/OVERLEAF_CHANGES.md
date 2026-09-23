# Overleaf surgical update (Section 6) — preserves your comments

Apply these THREE text edits in place, plus upload 4 figures. Everything not
listed here is unchanged, so comments on those parts survive. Do NOT replace the
whole section.

## Figures to upload (new)
figures_cmass/labelbug_count_vs_s8.png
figures_cmass/field_recovery_Om_cmass.png
figures_cmass/fisher_corrected_cmass.png
figures_cmass/crossfid_corrected_cmass.png
(These are now unused, ignore or delete: kl_bars_cmass.png,
kl_bars_crossfidelity_cmass.png, crossfid_newprotocol_cmass.png)

================================================================================
## EDIT 1  --  "Results: Does SR Match HR?" section
FIND the old "3. Cosmology, cross-fidelity (primary)" paragraph and everything
through the second cosmology table and its KL-bars figure (i.e. the block that
contains \label{tab:srs-crossfid} AND \label{tab:srs-kl} AND
\label{fig:srs-kl}). DELETE that whole block. REPLACE with:

--------------------------------------------------------------------------------
\paragraph{3. Cosmology.} The cosmology comparison in the original report used
mislabeled cosmologies, an indexing bug we describe and fix in
\S\ref{sec:srs-correction}, so those numbers are not shown here. With correct
labels a single box constrains $\Omega_m$ and $\sigma_8$; at the power spectrum
level the raw LR field is already HR-like, while at the field level an SR-trained
posterior transfers to HR and an LR-trained one does not
(\S\ref{sec:srs-correction}, Table~\ref{tab:srs-corrected}).
--------------------------------------------------------------------------------

================================================================================
## EDIT 2  --  "Diagnostic: What is NOT Working" section
FIND the paragraph starting "\paragraph{Why the cosmology gain is modest.}"
(it quotes the LR KL as 0.0047). REPLACE the whole paragraph with:

--------------------------------------------------------------------------------
\paragraph{Where the cosmology value is.} At the power spectrum level the raw LR
field is already close to HR, since they share large-scale modes, so the power
spectrum is a forgiving summary with little headroom. The model's value is at the
field level (variance, voids, small-scale structure), which the power spectrum
summary is insensitive to. The corrected cross-fidelity in
\S\ref{sec:srs-correction} bears this out: SR and LR are indistinguishable at the
power spectrum level, but at the field level an SR-trained posterior transfers to
HR while an LR-trained one does not.
--------------------------------------------------------------------------------

================================================================================
## EDIT 3  --  "Follow-Up Results" section
KEEP unchanged (comments safe): the intro, "Held-out check: the bispectrum",
"Removing the power spectrum loss", and "Does padding help" (with their bispectrum
and protocolv2 figures).

DELETE these buggy blocks:
  (a) the "The real goal metric: does SR survive the jump to HR?" paragraph plus
      its cross-fidelity table (\label{tab:srs-crossfid2}) and the
      kl_bars_crossfidelity figure. [It sits between the intro and the bispectrum.]
  (b) the "The decisive test: does any of this help the goal metric?" paragraph
      plus its table (\label{tab:srs-goal-final}), the crossfid_newprotocol
      figure, and the "Three things come out of this" itemize.
  (c) if present in your Overleaf, the paragraphs "Reliability of the
      cross-fidelity comparison", "Why the summaries cannot separate SR from LR",
      "Field-level inference: also at the prior", and the old "Bottom line".

Then, right after the padding paragraph (ends "...training recipe, not the
training time, was the limiting factor."), PASTE this entire corrected subsection:

--------------------------------------------------------------------------------
\subsection{The Goal Metric: Corrected Cross-Fidelity Results}
\label{sec:srs-correction}

\paragraph{The bug.} After the analysis above, a routine physical sanity check
failed in a way that could not be real: the amplitude of the power spectrum had
essentially zero correlation with $\sigma_8$, and a fit of the power spectrum to
the five parameters explained only $0.2\%$ of the variance. The power spectrum
amplitude must scale with $\sigma_8$, so this pointed to a labeling error, not to
physics. Tracing it, we found that the processed count fields were written in
string-sorted simulation order ($0, 1, 10, 100, 1000, \dots$) while the cosmology
table is indexed numerically. Field number $k$ is therefore the halo field of
simulation string-sort$(k)$, not simulation $k$, so every field in the whole
pipeline was paired with the wrong cosmology. We proved this exactly: using the
original halo catalogs, the field halo count matches the catalog count under the
string-sort permutation for all $2000$ simulations, while the identity mapping
matches only the two fixed points. We corrected the cosmology table to field
order, backed up the old one, and patched the builder so it cannot regenerate the
wrong order.

\begin{figure}[h]
\centering
\includegraphics[width=0.85\linewidth]{figures_cmass/labelbug_count_vs_s8.png}
\caption{The indexing bug in one picture. Total halo count in a box against
$S_8 = \sigma_8\sqrt{\Omega_m/0.3}$. As the pipeline used the labels (left) the
correlation is zero; with the string-sort mapping corrected (right) the true
$0.93$ correlation reappears. Halo abundance must track $S_8$, so the left panel
was the signature of a labeling error, not of uninformative data.}
\label{fig:srs-labelbug}
\end{figure}

\paragraph{What this changes, and what it does not.} Every cosmology number we
computed before this point used the labels and is affected; the earlier
cross-fidelity tables have been removed and are replaced by the corrected results
below, and the earlier reading that the SR-versus-LR difference was an irreducible
coin flip was itself a product of the mislabeling. The results that do not use the
labels are unaffected and still stand: the power spectrum match of SR to HR, the
held-out bispectrum, the field-match statistics, the seam test, and the real space
error. The generator conditions on the cosmology as a style vector and was trained
with the scrambled labels, so its conditioning was uninformative during training;
we return to this at the end.

\paragraph{The data is informative after all.} With correct labels the picture
inverts. A halo count in this suite carries a strong cosmological signal: the
total halo count alone correlates with the combination $S_8 = \sigma_8
\sqrt{\Omega_m/0.3}$ at $0.93$ and with $\Omega_m$ at $0.85$. A Fisher forecast on
a single-box power spectrum now explains $98\%$ of the variance, where it
explained $0.2\%$ before, and it constrains $\Omega_m$ and $\sigma_8$ well, with
$\Omega_b$, $h$ and $n_s$ weak or unconstrained, which is the textbook pattern for
a low-redshift halo field. The field-level network tells the same story: after the
fix its predicted $\Omega_m$ correlates with the truth at $0.90$ on both training
and test boxes, where before the fix the correlation was zero and the output was a
constant. So the earlier conclusion that a single box carries almost no cosmology
was an artifact of the labeling, not a property of the data.

\begin{figure}[h]
\centering
\includegraphics[width=0.82\linewidth]{figures_cmass/field_recovery_Om_cmass.png}
\caption{Field-level CNN recovery of $\Omega_m$ on HR test boxes, before and after
the label fix. Before (left) the prediction is flat and uncorrelated with the
truth (the network returns a near-constant); after (right) it tracks the diagonal
at a correlation of about $0.90$, on both training and test boxes. Same
architecture, same data, only the labels changed.}
\label{fig:srs-recovery}
\end{figure}

\begin{figure}[h]
\centering
\includegraphics[width=0.72\linewidth]{figures_cmass/fisher_corrected_cmass.png}
\caption{Fisher forecast for a single-box power spectrum with corrected labels:
posterior width divided by prior width per parameter (lower means better
constrained, $1$ means no information). $\Omega_m$ is well constrained and
$\sigma_8$ is constrained, while $\Omega_b$, $h$ and $n_s$ are weak, the textbook
pattern for a low-redshift halo field. Two emulators (linear average slope and
quadratic at the prior centre) bracket the estimate.}
\label{fig:srs-fisher}
\end{figure}

\paragraph{Corrected cross-fidelity, and where SR helps.} We re-ran the goal
metric with correct labels at two levels, the power spectrum summary and the full
field. We train the posterior on one source, apply it to HR, and compare to a
posterior trained on HR, using an ensemble of independently retrained estimators
so the numbers are not single draws. Table~\ref{tab:srs-corrected} gives the
result.
\begin{table}[h]
\centering
\caption{Corrected cross-fidelity KL to HR (posterior trained on the named source,
evaluated on HR), lower is better, with the HR-to-HR floor for reference. Summary
uses the power spectrum; field uses a three-dimensional CNN posterior on the full
box.}
\label{tab:srs-corrected}
\begin{tabular}{lccc}
\toprule
level & LR (raw) & SR & HR floor \\
\midrule
summary $P(k)$ & $0.032$ & $0.088$ & $0.028$ \\
field (CNN)    & $0.49 \pm 0.81$ & $\mathbf{0.024}$ & $0.022$ \\
\bottomrule
\end{tabular}
\end{table}
\begin{figure}[h]
\centering
\includegraphics[width=0.7\linewidth]{figures_cmass/crossfid_corrected_cmass.png}
\caption{Corrected cross-fidelity KL to HR (log scale, lower is better) at the two
levels, with the HR-to-HR floor shown as the dashed line. At the summary level the
raw LR field is already at the floor and SR does not help. At the field level, an
LR-trained posterior does not transfer to HR (large and unstable), while an
SR-trained posterior sits at the floor. SR helps exactly where it corrects the
field.}
\label{fig:srs-crossfid-corrected}
\end{figure}
The two levels point in opposite directions, and both make sense. At the summary
level the raw LR power spectrum is already at the HR floor, so an LR-trained
posterior transfers to HR as well as an HR-trained one does, and there is no gap
for SR to close; SR even hurts a little, because the generator reshapes the
small-scale power in a way that does not match how HR responds to cosmology. At
the field level the raw LR box looks different enough from HR that an LR-trained
posterior does not transfer, and it does so unreliably, sometimes acceptably and
sometimes catastrophically, which is why its number carries such a large spread.
SR corrects the field to look like HR, so an SR-trained posterior transfers to HR
reliably, at the floor. This is exactly the deployment case the project is built
on, train the inference model on cheap SR-corrected boxes and apply it to
expensive HR, and at the field level SR makes it work where raw LR does not.

\paragraph{Padding on the goal metric.} We ran the two padding variants through
the corrected cross-fidelity at the summary level as well
(Table~\ref{tab:srs-padding}). Among the SR models the ordering from the earlier
report survives: no padding is best on the goal metric, trim-aware padding is
worse, and trim-unaware is worst, so for the cosmology objective the no-padding
model is the one to use. All three sit above the raw LR control at this level,
consistent with LR already being HR-like on the power spectrum. One number does
change under the fix: the mislabeled data made trim-unaware look catastrophic (a
factor of several hundred above the others), while with correct labels it is the
weakest variant by about a factor of two, not a blow-up. Its real-space error
(Figure~\ref{fig:srs-protocolv2}) remains the clearest and most direct sign that
it is the least reliable model.
\begin{table}[h]
\centering
\caption{Corrected cross-fidelity KL to HR at the summary (power spectrum) level
for the SR padding variants, with the LR control and the HR floor. Lower is
better. Field-level padding numbers are future work.}
\label{tab:srs-padding}
\begin{tabular}{lc}
\toprule
model & summary cross-fidelity KL \\
\midrule
SR, no padding & $\mathbf{0.087}$ \\
SR, trim-aware padding & $0.127$ \\
SR, trim-unaware padding & $0.172$ \\
\midrule
LR (control) & $0.031$ \\
HR floor & $0.028$ \\
\bottomrule
\end{tabular}
\end{table}

\paragraph{Bottom line.} With the labels fixed, the cosmology question has a
clear answer. The data constrains $\Omega_m$ and $\sigma_8$ from a single box. At
the power spectrum level the raw LR field is already HR-like, so SR is not needed
there and slightly hurts. At the field level, which is what SR actually corrects,
an SR-trained posterior transfers to HR at the floor while a raw LR one does not,
so SR provides a real and reliable cosmology benefit exactly where it operates.
Trim-unaware padding is still the weakest variant, worst on the goal metric and
on real-space error, though the label fix shrinks its apparent failure from
catastrophic to about a factor of two. Two honest caveats remain: the
field-level LR number has a wide spread and rests on a handful of retrained
networks, so the size of the gap should be firmed up with more seeds; and because
the generator was trained with the scrambled labels, its cosmology conditioning
was uninformative, so a retrain with correct conditioning may sharpen the SR
result further. Neither caveat changes the direction: with correct labels, SR
helps cosmology at the field level.
--------------------------------------------------------------------------------

(The "A data reliability note" paragraph and the "Files" list after this are
unchanged.)
