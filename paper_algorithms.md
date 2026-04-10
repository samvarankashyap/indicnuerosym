# Constrained Decoding Algorithms

## Algorithm 1: Dead State Detection via Arithmetic Bounds

```
Algorithm 1: IsReachable(B, n)
────────────────────────────────────────────────────────────────
Input:  B = set of active NFA branches,
        n = syllables consumed so far in current line
Output: True if any branch can reach ACCEPT with total
        syllables in [N_min, N_max] (= [11, 15] for Dwipada)

 1  for each branch (s, g, j, M) in B do
 2      if s = ACCEPT then
 3          if n in {11, 12, 13, 14, 15} then return True
 4          continue
 5      end if
 6      P <- pattern of gana g from pool(s)      // Indra if s <= 2, Surya if s = 3
 7      r <- |P| - j                             // symbols remaining in current gana
 8      if s <= 2 then                           // still in Indra slots
 9          lo <- r + (2 - s) * 3 + 2            // min: remaining Indra ganas (3 syl each) + Surya (2 syl)
10          hi <- r + (2 - s) * 4 + 3            // max: remaining Indra ganas (4 syl each) + Surya (3 syl)
11      else                                     // in Surya slot (s = 3)
12          lo <- r
13          hi <- r
14      end if
15      if n + lo <= N_max  AND  n + hi >= N_min then
16          return True                           // this branch can land in [11, 15]
17      end if
18  end for
19  return False                                  // all branches doomed
```

**Complexity:** O(|B|) per call, O(1) per branch. No precomputation required.

**Notation:**
- Branch tuple (s, g, j, M): slot index s in {0,1,2,3,ACCEPT}, gana name g, sub-position j within gana pattern, matched ganas history M
- pool(s): INDRA_GANAS if s <= 2, SURYA_GANAS if s = 3
- N_min = 11, N_max = 15 (valid Dwipada line lengths)

---

## Algorithm 2: Constrained Decoding via Rejection Sampling

```
Algorithm 2: RejectionSamplingGeneration(theta, T, tau, K, p)
────────────────────────────────────────────────────────────────
Input:  theta = language model, T = tokenizer,
        tau = temperature, K = top-K search width,
        p = nucleus sampling threshold
Output: Two-line Dwipada poem text

 1  ids <- []                                    // generated token IDs
 2  lines_done <- 0
 3  while lines_done < 2 do
 4      z <- theta.forward(ids)                  // logits for next position
 5      pi <- softmax(z / tau)                   // token probabilities
 6      Sort candidates by pi in descending order
 7
 8      // ---- Stateless full-text analysis ----
 9      text <- T.decode(ids)
10      state <- AnalyzeText(StripPrefix(text))  // fresh FST + NFA on full text
11      lines_done <- state.lines_complete
12      if lines_done >= 2 then break
13
14      // ---- Force newline on valid line completion ----
15      if state.has_accept AND state.syl_count in {11..15} then
16          ids <- ids || [NEWLINE_TOKEN]
17          continue
18      end if
19
20      // ---- Force newline on overshoot (fallback) ----
21      if state.syl_count > 15 AND NOT state.has_accept then
22          ids <- ids || [NEWLINE_TOKEN]
23          continue
24      end if
25
26      // ---- Widen search near completion zone ----
27      K' <- K
28      if state.syl_count >= 8 then
29          K' <- 2K
30      end if
31
32      // ---- Rejection loop over top-K' candidates ----
33      V_accept <- {}                            // candidates reaching ACCEPT
34      V_alive  <- {}                            // candidates keeping NFA alive
35      for k = 1 to K' do
36          c <- k-th most probable token
37          if pi(c) < 1e-8 then break
38          if c = EOS then
39              if lines_done >= 2 then select c; break
40              else continue
41          end if
42          if NOT IsTeluguToken(T.decode(c)) then continue
43
44          // Full re-analysis: decode all tokens so far + candidate
45          text' <- T.decode(ids || [c])
46          r <- AnalyzeText(StripPrefix(text'))
47
48          if NOT r.alive then continue           // NFA dead or unreachable
49          if "\n" in T.decode(c) AND r.lines_complete <= lines_done then continue
50          if r.syl_count > 15 AND NOT r.has_accept then continue
51
52          if r.has_accept AND r.syl_count in {11..15} then
53              V_accept <- V_accept + {(c, pi(c))}
54          end if
55          V_alive <- V_alive + {(c, pi(c))}
56
57          if |V_accept| >= 5 then break          // enough accept candidates
58      end for
59
60      // ---- Select from best candidate set ----
61      V <- V_accept if V_accept != {} else V_alive
62      if V != {} then
63          Apply nucleus filter (top-p) to V
64          chosen <- weighted multinomial sample from V
65      else
66          // Exhaustive fallback: scan up to 500 tokens
67          for k = 1 to 500 do
68              c <- k-th most probable token
69              if IsTeluguToken(T.decode(c)) then
70                  r <- AnalyzeText(StripPrefix(T.decode(ids || [c])))
71                  if r.alive AND (r.syl_count <= 15 OR r.has_accept) then
72                      chosen <- c; break
73                  end if
74              end if
75          end for
76      end if
77      ids <- ids || [chosen]
78  end while
79  return T.decode(ids)
```

**Key subroutine -- AnalyzeText(text):**

```
function AnalyzeText(text)
────────────────────────────────────────────────────────────────
    Instantiate fresh SyllableAssembler, GuruLaghuClassifier
    B <- SpawnSlot(0, ())                        // initial Gana NFA branches
    markers <- []
    for each item in SyllableAssembler.process(text) do
        if item = "\n" then
            flush classifier; advance B with flushed labels
            if any branch in B is ACCEPT then lines_complete++
            B <- SpawnSlot(0, ())                // reset for next line
            markers <- []
        else if item = " " then
            flush classifier boundary; advance B
        else
            feed syllable to classifier; advance B with emitted labels
        end if
    end for
    flush remaining; advance B
    syl_count <- |markers|
    has_accept <- any (ACCEPT, *, *, *) in B AND syl_count in {11..15}
    alive <- IsReachable(B, syl_count) OR lines_complete >= 2
    return {alive, lines_complete, has_accept, syl_count, B}
```

**Complexity:** O(K * L) per generation step, where L is the current sequence length (due to full re-analysis). Total: O(T_max * K * L_avg).

---

## Algorithm 3: Constrained Decoding via Logit Masking with Backtracking

```
Algorithm 3: MaskedGenerationWithBacktracking(theta, T, V_tel, M_static, tau, tau_max, Delta_tau, D)
────────────────────────────────────────────────────────────────
Input:  theta = language model, T = tokenizer,
        V_tel = (telugu_ids, telugu_texts) pre-extracted Telugu token set,
        M_static = static mask tensor (0 for Telugu+EOS, -inf elsewhere),
        tau = base temperature, tau_max = temperature ceiling (1.2),
        Delta_tau = temperature bump per backtrack (0.15),
        D = decay steps before temperature returns to base (6)
Output: Two-line Dwipada poem text

 1  ids <- [];  S <- new CompositeState()
 2  lines_done <- 0;  tau_curr <- tau
 3  C <- []                                      // checkpoint ring buffer (max 15)
 4  b <- 0;  b_consec <- 0;  d_remaining <- 0    // backtrack counters
 5
 6  for t = 1 to T_max do
 7      if lines_done >= 2 then break
 8
 9      // ---- Forward pass with static masking ----
10      z <- theta.forward(ids) / tau_curr
11      z <- z + M_static                         // kill all non-Telugu tokens
12
13      // ---- Save checkpoint every 3 tokens ----
14      if t mod 3 = 0 then
15          C.append( (t, S.snapshot(), |ids|, lines_done, tau_curr) )
16          if |C| > 15 then C <- C[-15:]          // ring buffer
17      end if
18
19      // ---- Force newline on valid line completion ----
20      S_flush <- clone(S); S_flush.flush()
21      if S_flush.has_accept() then
22          z <- -inf everywhere; z[NEWLINE] <- 0
23          S.feed_char("\n")
24          lines_done <- S.lines_complete
25          ids <- ids || [NEWLINE_TOKEN]
26          b_consec <- 0
27          continue
28      end if
29
30      // ---- Build dynamic Gana NFA mask ----
31      snap <- S.snapshot()
32      V_valid <- BuildGanaMask(snap, V_tel)     // Algorithm 5
33
34      // ---- Filter spurious newlines ----
35      if NEWLINE in V_valid then
36          S' <- clone(S); S'.feed_char("\n")
37          if S'.lines_complete <= lines_done then
38              V_valid <- V_valid \ {NEWLINE}
39          end if
40      end if
41
42      // ---- Backtrack trigger ----
43      need_backtrack <- (|V_valid| < 3) OR (S_flush.syl_count > 15)
44      if need_backtrack AND |C| > 0 AND b < 30 then
45          b <- b + 1;  b_consec <- b_consec + 1
46
47          // Escalating pop: pop more checkpoints on consecutive backtracks
48          pop <- min(b_consec, |C|)
49          for i = 1 to pop do
50              ckpt <- C.pop()
51          end for
52          (t_ckpt, snap_ckpt, len_ckpt, lines_ckpt, tau_ckpt) <- ckpt
53
54          // Restore state
55          S <- CompositeState.from_snapshot(snap_ckpt)
56          ids <- ids[1 .. len_ckpt]
57          lines_done <- lines_ckpt
58          t <- t_ckpt
59
60          // Temperature escalation + reseed
61          tau_curr <- min(tau_ckpt + Delta_tau * b_consec,  tau_max)
62          d_remaining <- D
63          RNG.seed(seed + b * 1337)
64          continue
65      end if
66
67      // ---- Apply dynamic mask and sample ----
68      M_dyn <- -inf everywhere
69      for each tid in V_valid do M_dyn[tid] <- 0
70      z <- z + M_dyn
71      pi <- softmax(z)
72      chosen <- sample from pi with nucleus filtering (top-p)
73
74      // ---- Commit token ----
75      ids <- ids || [chosen]
76      S.feed_token_text(T.decode(chosen))
77      lines_done <- S.lines_complete
78
79      // ---- Temperature decay ----
80      b_consec <- 0
81      if d_remaining > 0 then
82          d_remaining <- d_remaining - 1
83          if d_remaining = 0 then tau_curr <- tau
84      end if
85  end for
86  return T.decode(ids)
```

**Key subroutine -- BuildGanaMask:**

```
Algorithm 5: BuildGanaMask(snap, V_tel)
────────────────────────────────────────────────────────────────
Input:  snap = CompositeState snapshot,
        V_tel = (telugu_ids, telugu_texts)
Output: V_valid = set of token IDs that keep all NFAs alive

 1  V_valid <- {}
 2  for each (tid, text) in zip(telugu_ids, telugu_texts) do
 3      S' <- CompositeState.from_snapshot(snap)
 4      S'.feed_token_text(text)
 5      S'.flush()
 6      if S'.is_alive() then                    // Gana reachable AND Prasa alive AND Yati alive
 7          V_valid <- V_valid + {tid}
 8      end if
 9  end for
10  return V_valid
```

**Complexity:** O(|V_tel|) per generation step for mask building, where |V_tel| ~ 1,822. Each clone + feed + flush is O(token_length). Total: O(T_max * |V_tel|). Independent of sequence length (incremental state).

---

## Algorithm 4: Hybrid Constrained Decoding (Mask + Rejection)

```
Algorithm 4: HybridGeneration(theta, T, V_tel, M_static, tau, K_rej)
────────────────────────────────────────────────────────────────
Input:  theta = language model, T = tokenizer,
        V_tel, M_static = pre-computed Telugu token data,
        tau = temperature, K_rej = rejection search width (100)
Output: Two-line Dwipada poem text

 1  ids <- [];  S <- new CompositeState()
 2  lines_done <- 0;  tau_curr <- tau
 3  C <- []                                      // checkpoint ring buffer
 4  b <- 0;  b_consec <- 0
 5
 6  for t = 1 to T_max do
 7      if lines_done >= 2 then break
 8
 9      // ---- Forward pass + static mask ----
10      z <- theta.forward(ids) / tau_curr
11      z <- z + M_static
12
13      // ---- Checkpoint (every 3 tokens) ----
14      if t mod 3 = 0 then
15          C.append( (t, S.snapshot(), |ids|, lines_done, tau_curr) )
16          if |C| > 15 then C <- C[-15:]
17      end if
18
19      // ---- Force newline on ACCEPT ----
20      S_flush <- clone(S); S_flush.flush()
21      if S_flush.has_accept() then
22          S.feed_char("\n")
23          lines_done <- S.lines_complete
24          ids <- ids || [NEWLINE_TOKEN]
25          b_consec <- 0;  continue
26      end if
27
28      // ---- Build + apply dynamic Gana NFA mask ----
29      snap <- S.snapshot()
30      V_valid <- BuildGanaMask(snap, V_tel)
31      // Filter spurious newlines
32      if NEWLINE in V_valid then
33          S' <- clone(S); S'.feed_char("\n")
34          if S'.lines_complete <= lines_done then
35              V_valid <- V_valid \ {NEWLINE}
36          end if
37      end if
38      M_dyn <- -inf everywhere
39      for each tid in V_valid do M_dyn[tid] <- 0
40      z <- z + M_dyn
41
42      // ---- NFA rejection over top-K from masked distribution ----
43      pi <- softmax(z)
44      Sort candidates by pi descending
45      V_accept <- {};  V_alive <- {}
46
47      for k = 1 to K_rej do
48          c <- k-th candidate
49          if pi(c) < 1e-8 then break
50          if c = EOS then
51              if lines_done >= 2 then select c; break
52              else continue
53          end if
54
55          // Clone composite state, simulate feeding candidate token
56          S' <- CompositeState.from_snapshot(snap)
57          for each ch in T.decode(c) do S'.feed_char(ch)
58          S'_flush <- clone(S'); S'_flush.flush()
59
60          if NOT S'_flush.is_alive() then continue
61          if "\n" in T.decode(c) AND S'.lines_complete <= lines_done then continue
62          if S'_flush.syl_count > 15 AND NOT S'_flush.has_accept() then continue
63
64          if S'_flush.has_accept() then
65              V_accept <- V_accept + {(c, pi(c))}
66          end if
67          V_alive <- V_alive + {(c, pi(c))}
68
69          // Early termination
70          if |V_accept| >= 3 then break
71          if |V_alive| >= 10 AND V_accept = {} then break
72      end for
73
74      // ---- Select from best set ----
75      V <- V_accept if V_accept != {} else V_alive
76      if V != {} then
77          Normalize probabilities in V
78          chosen <- weighted multinomial sample from V
79      else
80          // ---- Backtrack (rare fallback) ----
81          if |C| > 0 AND b < 30 then
82              b <- b + 1;  b_consec <- b_consec + 1
83              pop <- min(b_consec, |C|)
84              for i = 1 to pop do ckpt <- C.pop()
85              Restore S, ids, lines_done, t from ckpt
86              tau_curr <- min(ckpt.tau + 0.15 * b_consec, 1.2)
87              RNG.seed(seed + b * 1337)
88              continue
89          else
90              chosen <- most probable token       // last resort
91          end if
92      end if
93
94      // ---- Commit ----
95      ids <- ids || [chosen]
96      S.feed_token_text(T.decode(chosen))
97      lines_done <- S.lines_complete
98      b_consec <- 0
99
100     // Temperature decay
101     if d_remaining > 0 then
102         d_remaining <- d_remaining - 1
103         if d_remaining = 0 then tau_curr <- tau
104     end if
105 end for
106 return T.decode(ids)
```

**Complexity:** O(|V_tel| + K_rej) per step. The mask building dominates at O(|V_tel|) ~ O(1,822). The rejection loop is bounded by K_rej = 100. Total: O(T_max * |V_tel|). Backtracking is rare (<5% of generations in practice).

---

## Comparison of Strategies

| Property | Rejection (Alg 2) | Masking+Backtrack (Alg 3) | Hybrid (Alg 4) |
|----------|-------------------|---------------------------|----------------|
| State tracking | Stateless (re-analyze full text) | Incremental (CompositeState) | Incremental (CompositeState) |
| Per-step cost | O(K * L) | O(\|V_tel\|) | O(\|V_tel\| + K_rej) |
| Depends on seq length | Yes (L grows) | No | No |
| Coverage | Complete (tests all top-K) | Complete (tests all Telugu tokens) | Complete (mask) + selective (reject top-K_rej) |
| Dead-end recovery | Exhaustive fallback (500 tokens) | Checkpoint backtracking | Checkpoint backtracking (rare) |
| Temperature adaptation | None | Escalation (+0.15/backtrack, cap 1.2, decay over 6 steps) | Escalation (same) |
| Constraints enforced | Gana only | Gana + Prasa + Yati | Gana + Prasa + Yati |
| Line completion | Force newline on ACCEPT | Force newline on ACCEPT | Force newline on ACCEPT, prefer accept candidates |

---

## Notation Reference

| Symbol | Meaning |
|--------|---------|
| theta | Language model (Gemma 3 1B) |
| T | Tokenizer |
| z | Raw logit vector |
| pi | Probability distribution (after softmax) |
| tau | Sampling temperature |
| K | Top-K search width |
| p | Nucleus (top-p) threshold |
| ids | Sequence of generated token IDs |
| S | CompositeState (incremental FST+NFA pipeline state) |
| B | Set of active NFA branches |
| (s, g, j, M) | Branch: slot, gana name, sub-position, matched history |
| V_tel | Pre-extracted Telugu token set (~1,822 tokens) |
| M_static | Static logit mask: 0 for Telugu/EOS, -inf elsewhere |
| M_dyn | Dynamic logit mask from Gana NFA alive check |
| N_min, N_max | Valid line length bounds (11, 15) |
| C | Checkpoint ring buffer |
| b, b_consec | Total and consecutive backtrack counters |
| Delta_tau | Temperature bump per backtrack (0.15) |
| D | Temperature decay window (6 steps) |
