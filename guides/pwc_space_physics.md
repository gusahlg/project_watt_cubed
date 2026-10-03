# PWC: matter, gravity, curved voxels, and a world that remains buildable

**A critical architecture study for Gustav Ahlgren**  
**3 October 2026 · Proposed design, not an implementation specification or a benchmark**

This document investigates how Project Watt Cubed could support enormous rounded worlds, space, matter-derived gravity, strong artificial structures, and eventually planetary deformation and fragmentation. It deliberately revises several claims in the preceding conversation.

The recommendation is to make **matter and its physical state authoritative, represent it through persistent local voxel patches, derive gravity from mass, and treat physical deformation and changes of voxel layout as separate operations**. A planet need not be a privileged engine object. A coordinate patch, a mechanical connection, and a conserved quantity do need precise definitions.

## 0. Short report: the most important findings

| Issue | What the investigation found | Consequence for PWC |
|---|---|---|
| Small curvature was confused with small distortion | A huge radius makes adjacent surface normals almost parallel. It does not eliminate the stretching and skew of a particular cube-to-sphere projection. | Judge the actual cell geometry, including face corners and deep interiors. |
| Three operations were conflated | Relabelling coordinates, moving matter, and aligning a building grid with gravity are different problems. | Give them separate state and algorithms. |
| Rounding matter does not align its blocks | An elastic or plastic solver can round an object while leaving its internal grid stretched or poorly aligned. | A mechanics solver alone does not deliver the desired building experience. |
| The earlier topology claim was too broad | A cube-shaped solid can be deformed into a rounded solid. A global map can also deform several already-disconnected regions. The real restrictions concern grid alignment, injectivity, and changes of material connectivity. | Do not justify every domain boundary with an incorrect impossibility claim. |
| Gravity magnitude is insufficient | Strong nearly uniform gravity makes an unsupported object fall; support forces, self-gravity, and tides determine its internal loading. | Solve or approximate mechanics, rather than assigning a scalar warp from gravity strength. |
| The center problem survives the stress correction | Removing hydrostatic visual compression cannot prevent radial voxel columns from converging at a center. | Interior layout requires its own solution. |
| Material mass is still an architectural decision | The existing reaction proposal conserves constituent count, not an established physical mass or energy. | Define conserved matter amounts before deriving gravity from material identity. |
| A smooth displacement field cannot perform fracture | Continuous deformation preserves connectivity until the representation is changed or becomes invalid. | Fracture must remove connections and permit separate motion explicitly. |
| Slow interpolation is not automatically safe | Two valid endpoint geometries can have an invalid straight interpolation between them. | Validate the trajectory, including collision and local inversion. |
| Procedural gravity is not free | Arbitrary terrain noise cannot generally supply accurate mass integrals without work. Refinement must also avoid double counting. | World generation needs a deliberate mass-summary contract. |
| “Infinite universe” needs a gravity boundary rule | An arbitrary infinite mass distribution does not automatically define a unique, convergent Newtonian field. | Specify the universe-scale law instead of leaving a hidden cutoff in the implementation. |
| Multiplayer state becomes richer | Seed plus block edits cannot reconstruct deformations, broken bonds, moving assemblies, or remapping history by itself. | Persist and replicate authoritative geometry and topology changes. |

**My assessment:** the overall direction fits PWC. The single universal gravity-driven warp proposed earlier is not a sufficient foundation. The strongest version is a generic system of material amounts, local cell layouts, physical embeddings, and mechanical connections. It accepts limited geometric irregularity and explicit topology changes.

**The biggest unresolved product choice:** whether a block is fundamentally an everlasting individually shaped object, or a cell containing matter that can be reorganized during major physical change. Perfectly permanent cubes, arbitrary reshaping, globally smooth gravity alignment, and an ordinary six-neighbor integer grid everywhere cannot all be promised together.

**The first experiment should test geometry and building, before committing to a planetary mechanics rewrite.** The gravitational force calculation is comparatively well understood. Pleasant, stable, editable curved voxel geometry is the higher architectural risk.

## How to read this document

Sections 1–5 establish the philosophy and the representation. Sections 6–11 develop gravity, materials, mechanics, geometric validity, and topology changes. Sections 12–16 cover gameplay, multiplayer, performance, and code organization. Sections 17–19 give a staged plan, adversarial tests, and remaining decisions.

Quick navigation: [issues](#0-short-report-the-most-important-findings) · [architecture](#4-the-recommended-architecture-and-its-alternatives) · [gravity](#6-gravity-derived-from-matter) · [mechanics](#8-mechanics-how-matter-should-actually-move) · [implementation plan](#17-a-staged-implementation-plan-with-exit-criteria) · [sources](#appendix-b-sources-and-how-they-were-used).

Appendix A contains independent calculations behind the main corrections. Appendix B is an annotated source list. Appendix C records what was and was not verified.

Throughout:

- **Established result** means mathematics or physics with a stated domain of applicability.
- **Engineering recommendation** means my proposed design for PWC.
- **Research risk** means the requested combination still needs a convincing prototype.
- Numerical tolerances and budgets are starting proposals unless explicitly identified as calculations.

## 1. What this should preserve about PWC

### 1.1 The philosophy recovered from your own statements

Your recurring priorities are a small capable core, freedom arising from reusable rules, meaningful building and experimentation, community and multiplayer, and deep modding. You have described PWC as a platform for an unusually creative sandbox, rather than a game organized around a prescribed objective.

Your material discussions add a more specific preference: new behavior should arise from configurations and their relationships, without requiring a hand-authored category for every material or outcome.

I interpret those commitments as the following architecture principles:

1. **Mechanisms should apply to unfamiliar constructions.** A player-made object should not lose a capability because nobody labelled it a planet.
2. **The core should expose composable rules.** Gravity, occupancy, contact, material state, and coordinate representation should be useful independently.
3. **Player engineering should have understandable consequences.** A surprising result should be traceable to a rule, rather than an invisible classification change.
4. **Representation should not dictate fictional physics accidentally.** Moving a chunk boundary or changing a graphics setting must not change material strength.
5. **Performance belongs in the design.** A theoretically uniform rule that needs every voxel in the universe to be active is not useful universality.
6. **Building deserves the same priority as simulation.** An impressive planet solver that makes walls, machinery, and movement frustrating fails the purpose of the game.

These are interpretations of your stated priorities, not additional decisions you have already approved.

### 1.2 Evidence from the current project

The public README describes a procedural world with sparse edit replication and a headless authoritative server. It also describes a separate Vulkan renderer. The Cargo manifest exposes a material workspace crate and a sibling `voxel-engine` dependency. Those are useful constraints: new authoritative physics should be usable without a GPU, and renderer changes should respect the existing separation. [S1], [S2]

The saved reaction specification is explicitly a candidate. It changes resource configurations while preserving the constituent count of each interacting block. It says that mechanics, transport, destruction, optical behavior, and physical energy accounting require additional rules. Its resource coordinates do not already mean hardness or density.

I inspected those documents and the available public README and manifest. I did **not** complete a source-code audit of the game or renderer. Directory-level proposals below are therefore suggested destinations, not claims that those modules already exist.

### 1.3 Removing names is not enough

Replacing `Planet` with `MassDomain` achieves little if `MassDomain` secretly means “find a big round object and give it special spherical behavior.”

The important test is operational:

> Could the same state and interfaces describe a stone, a weak aggregate, a rotating station, a split world, and an enormous reinforced cube?

A generic `Assembly` is justified if it describes material whose motion is mechanically coupled. A `Patch` is justified if it describes a locally structured coordinate map. Neither needs to confer a special gravitational law.

Conversely, a content generator may still contain a function named `generate_starting_planet`. A content function that arranges ordinary matter is compatible with a generic simulation. What would violate your aim is permanent gravitational privilege attached to the name.

### 1.4 A universal law still contains authored choices

Choosing the force law, material stiffness, a plastic flow model, a mesh-quality objective, or the amount of permissible distortion is designing the universe.

There is no completely assumption-free “natural warp.” The useful standard is that the assumptions are few, explicit, consistent, and open to experimentation.

This also suggests restraint with realism. Newtonian dynamics and a simple constitutive model may be valuable because they connect many phenomena. Accurately reproducing every property of rock is a different objective.

## 2. The promises that can and cannot coexist

### 2.1 Separate the desired outcomes

The original request contains several goals:

- Globally round worlds with locally comfortable block building.
- Gravity produced by actual matter, including ungenerated procedural matter.
- Small objects that remain recognizably blocky.
- Material-dependent resistance to deformation.
- Freedom to create, demolish, split, or reshape enormous structures.
- Accessible interiors, including regions near a center.
- Stable multiplayer behavior and persistent constructions.
- A sufficiently small and efficient core.

These goals are compatible at the level of a platform. They are not all satisfied by one smooth map from a permanent global Cartesian grid.

### 2.2 A priority order I recommend

**Hard correctness requirements**

- Matter is not duplicated or silently lost.
- Occupied geometry does not acquire unintended overlap or inversion.
- Every authoritative edit has an unambiguous target and revision.
- Rendering, selection, collision, and material accounting describe the same world.
- Loading and unloading do not change the physical law.
- A representation update is distinguishable from physical motion.

**Strong gameplay goals**

- Ordinary construction areas have nearly cubic cells.
- Local movement is comfortable and metrically consistent.
- Existing structures retain useful identity and attachment information.
- Geological change is gradual or physically event-driven.

**Approximate goals**

- Block faces align with gravity where a useful layout permits it.
- Material relaxation resembles plausible mechanics.
- Spatial and temporal approximation errors stay within declared budgets.
- Exceptional topology occupies limited, inspectable regions.

**Promises to abandon**

- Every block remains an identical perfect Euclidean cube.
- Every cell always has exactly one same-sized neighbor on each of six faces.
- Every material cell keeps the same simple global integer address through arbitrary remeshing.
- Any gravity field can be integrated into a globally regular aligned voxel grid.

This ordering is a recommendation. If permanent block identity matters more than large-scale reshaping, the architecture should favor intact deforming blocks and accept greater geometric distortion.

### 2.3 The actual geometric obstruction

A spherical surface cannot be covered by an everywhere-regular square lattice with the connectivity of an infinite flat grid. For a closed quadrilateral mesh of a sphere, Euler's formula gives

$$
\sum_v(4-d_v)=8,
$$

where `d_v` is the number of incident edges at vertex `v`. A cube-sphere realizes this with eight vertices of valence three.

Those defects can be organized and made unobtrusive. They cannot be eliminated merely by making the planet bigger.

This is different from claiming that a solid cube cannot become a ball. Both solids have the same basic topology. Likewise, a deformation of ambient space can affect two disconnected objects differently without requiring a separate universe for each.

The restrictions arise when we additionally demand low distortion, gravity alignment, globally regular grid connectivity, and topology-changing material events. Appendix A derives the simple cases.

Research on hexahedral meshes makes the distinction explicit: local frame directions, a compatible coordinate map, and a valid cell mesh are separate products. An attractive direction field is not enough. [S5], [S6]

## 3. Corrections to the earlier proposals

### 3.1 A large planet does not rescue every cube-sphere projection

For a spherical surface of radius `R`, a physical displacement `L` rotates the normal by approximately

$$
\Delta\theta \approx \frac{L}{R}.
$$

That is a curvature statement.

Now examine the normalized face projection

$$
p(u,v)=R\frac{(u,v,1)}{\sqrt{1+u^2+v^2}},
\qquad -1\le u,v\le 1.
$$

Its two tangent directions have angle `α` satisfying

$$
\cos\alpha=-\frac{uv}{\sqrt{(1+u^2)(1+v^2)}}.
$$

At the face center they meet at 90°. At a face corner they meet at 120°. **The radius cancels.**

Making the planet a million times larger makes each cell's curvature smaller, but this local skew remains. For this specific projection, equal parameter steps also have very different physical lengths near the corners.

Other mappings redistribute the distortion. They trade orthogonality, cell area, edge length, and seam behavior. Cubed-sphere research discusses these tradeoffs; it does not provide an everywhere-identical cubic lattice. [S3], [S4]

**Recommendation:** retain the cube-sphere as a geometry prototype, not as proof that the problem has been solved.

### 3.2 Physical relaxation does not solve the building-grid problem

Suppose a giant cube of weak material relaxes into a rounder mass.

If the original voxel coordinates stay attached to its material, some rows stretch, others compress, and their orientation records the deformation history. Nothing in ordinary elasticity requires the final rows to become nice radial columns with horizontal block faces.

If, instead, the engine reorganizes those rows into a convenient new grid, it has performed an additional operation: a change of material representation, possibly involving cells being split or merged.

A successful architecture must describe both operations and the information transferred between them.

### 3.3 Gravity gradients are not the whole explanation of stress

The previous answer correctly rejected `warp ∝ |g|`. But “only differences in gravitational acceleration deform objects” needs a qualification.

A freely falling small object in an approximately uniform gravitational field has little internal loading from that field. The same object resting on supports can experience significant stress: its weight must travel through its structure to those supports.

A bridge in a nearly uniform field can bend. A hanging cable can be under tension. Neither requires a large tidal gradient.

The governing ingredients are body forces, acceleration, support and contact forces, material response, and prior state. Standard solid mechanics expresses this through a momentum balance, rather than a local function of gravity magnitude. [S10]

### 3.4 The center has two independent problems

**Physical problem:** deep material can be under large pressure even where net gravity is zero.

**Representation problem:** a fixed set of angular columns becomes narrower as radius decreases. All radial directions meet at the center.

Suppressing pressure-driven volume change affects the first problem. It does not remove the second.

A finite Cartesian or otherwise non-radial core is therefore a sensible representation. It does not need weaker material or artificial gravity. The grid simply stops trying to align with an undefined “up” direction.

### 3.5 A deformation field cannot silently tear

For a continuous one-to-one map, the image of a connected solid remains connected. A crack therefore requires a change of material connectivity, a discontinuity between independently moving pieces, or a new representation of the material.

A low-resolution field that remains connected across a newly mined gap can produce invisible elastic bridges. A scalar occupancy average does not reliably detect that error.

**Recommendation:** store mechanical connections explicitly enough that removing matter can remove load paths. Do not use gravity wells to decide whether two surfaces are welded.

### 3.6 Slow motion is not a validity guarantee

Consider two completely valid maps:

$$
\chi_0(X)=X,
\qquad
\chi_1(X)=
\begin{pmatrix}
-1&0&0\\
0&-1&0\\
0&0&1
\end{pmatrix}X.
$$

The second is a 180° rotation. Both determinants are positive.

Halfway through linear interpolation of their positions, the transformation becomes

$$
\frac12(D\chi_0+D\chi_1)=
\operatorname{diag}(0,0,1).
$$

The object has collapsed into a line.

Interpolating for several hours instead of several seconds does not repair this. Rigid motion needs rotation interpolation; deforming motion needs validated increments and swept collision checks.

### 3.7 “Smooth the cluster” is not yet a physical model

A surface filter can make a shape look rounder without conserving volume, respecting supports, maintaining cavities, or preserving material identity.

Even a volume-preserving filter is not automatically gravity-driven mechanics. It may round an unsupported beam, shrink a tunnel, or erase a deliberate structure for aesthetic reasons.

Surface fairing is a useful geometry-processing tool in its own right. It should not be presented as the missing gravitational law. [S19]

### 3.8 An enormous beam should not be exempt merely for being a beam

The earlier suggestion to suppress warping for low-compactness objects was a useful heuristic for choosing a spherical coordinate layout. It is not a general mechanical law.

A sufficiently large, weak beam can bend or collapse under self-gravity. A strong hollow structure may survive. A rapidly rotating body can take a non-spherical equilibrium shape.

Compactness may guide a mesher or a numerical approximation. Material forces should determine mechanical behavior.

## 4. The recommended architecture and its alternatives

### 4.1 Compare the actual choices

| Representation | Main advantage | Main cost | Suitability |
|---|---|---|---|
| One Cartesian lattice with spherical occupancy | Simple addresses and ordinary chunks | Sloping/stair-stepped block surfaces relative to radial gravity | Good gravity prototype; poor match for the final building goal |
| One permanent cube-sphere shell | Convenient radial “up” over much of a surface | Projection distortion, seams, depth and altitude limits | Strong constrained prototype |
| One universal smooth displacement field | Simple conceptual formula | No automatic fracture or retiling; distortion can accumulate | Useful within a fixed deformable region |
| Arbitrary gravity-aligned hexahedral meshing everywhere | Closest to the ideal geometric goal | Difficult validity, connectivity, persistence, and runtime cost | Research path, not an initial dependency |
| Material particles with MPM | Large flow and topology changes are natural to the representation | Persistent block faces and exact construction identity become harder | Useful possible backend for loose or flowing matter |
| Persistent local voxel patches with separate mechanics and explicit topology events | Preserves efficient local grids while exposing the difficult boundaries | Requires careful interfaces and some exceptional geometry | Recommended platform direction |

### 4.2 The target in one paragraph

PWC stores matter in locally structured patches. Each patch has a reference layout and a current physical embedding. Mechanical connections determine whether material moves together, deforms, separates, or comes into contact. A mass hierarchy describes that matter in physical space and provides gravitational queries. Coordinate-layout algorithms can construct or improve patches, but they cannot silently change physical matter. Fracture and major reorganization use explicit transactions that conserve quantities and update references.

There is no required `Planet` type. A round world is one possible configuration of these generic primitives.

### 4.3 What this recommendation does not establish

It does not prove that a fully automatic, arbitrarily deforming, gravity-aligned voxel world will meet PWC's performance and usability requirements.

It creates places in the architecture where those questions can be tested without contaminating gravity, material identity, networking, or ordinary chunks.

The recommended first playable version has stable generated curved terrain and moving local assemblies. Large-scale yielding and automatic retiling are later capabilities with independent acceptance gates.

That distinction matters: stable curved worlds are an achievable implementation target; the unrestricted version is a research program.

## 5. Define state before choosing solvers

### 5.1 Four spaces with different meanings

| Space | Purpose | Example |
|---|---|---|
| Resource/configuration space | What the material is | A configuration of resource-coordinate tuples |
| Patch coordinates | Where a cell is stored locally | Patch 41, chunk (8, −2, 0), cell (3, 7, 1) |
| Reference material geometry | Physical shape before subsequent deformation | A curved cell's initial vertices and volume |
| Current physical space | Where matter and entities are now | Position relative to an assembly and a universe sector |

Resource coordinates do not become world coordinates. A changed material configuration need not move its containing cell.

A patch coordinate is an address, not automatically a meter measurement.

### 5.2 A mapping with explicit responsibilities

Let `q` be a local coordinate in a patch. Let `X_p(q)` define the patch's reference physical geometry. Let `φ_a(X,t)` describe material deformation in assembly `a`. Let `R_a(t)` and `c_a(t)` describe its rigid pose.

Then

$$
x(q,t)=c_a(t)+R_a(t)\,\phi_a(X_p(q),t).
$$

For undeformed material, `φ_a(X,t)=X`.

A Cartesian patch can have an affine `X_p`. A curved patch can have a nonlinear `X_p`. Both use the same physical deformation interface.

The distinctions are important:

- The patch may be curved without being elastically strained.
- A rigid rotation is not material deformation.
- A new coordinate description can preserve every physical point.
- A physical deformation changes points while material labels remain attached.

For a reference Jacobian `J₀ = ∂X_p/∂q` and current assembly-local Jacobian `J = ∂(φ_a ∘ X_p)/∂q`, the material deformation gradient is

$$
F=J J_0^{-1}.
$$

Using `J` alone as elastic strain would mistakenly treat the original curved layout as a stressed material. Reference and current configurations are standard mechanics concepts; this particular layering is the proposed PWC design. [S9]

### 5.3 Addresses, handles, and lifetime

A possible address is:

~~~rust
struct CellAddress {
    patch: PatchId,
    chunk: I64Vec3,
    local: U16Vec3,
}

struct CellHandle {
    address: CellAddress,
    generation: u32,
}
~~~

These are illustrative shapes, not final memory layouts.

The generation prevents a stale reference from silently referring to a different cell after replacement.

Most ordinary block edits keep an address. Physical motion also keeps an address. A representation change may require a mapping from old cells to new cells, including one-to-many or many-to-one mappings.

Use stable entity IDs for machines and important attachments. They should not depend exclusively on a bare cell coordinate that might cease to exist.

### 5.4 Three different notions of neighbor

PWC should distinguish:

1. **Storage adjacency:** nearby cells in the same structured grid or across a patch interface.
2. **Mechanical connection:** material joined by a bond, joint, weld, or continuous solid.
3. **Physical contact:** surfaces currently touching in physical space.

They often coincide in ordinary terrain. They need not coincide on a fracture, moving ship, patch boundary, or contact between two separate bodies.

The gravitational hierarchy is a fourth structure. Its grouping exists to approximate forces, not to declare any of those relationships.

### 5.5 Avoid allocating a huge object per voxel

The conceptual model does not require a heap allocation, rigid body, or physics element for every block.

Ordinary chunks should retain dense or compressed material storage. Most cells share a patch mapping and a small number of mechanical summary records. Individual metadata is allocated only for edits, special attachments, damage detail, or local refinement.

The goal is to preserve a fast ordinary-grid path and make unusual boundaries explicit.

## 6. Gravity derived from matter

### 6.1 Use the physical mass distribution

For a finite isolated distribution, Newtonian gravity is described by

$$
\nabla^2\Phi=4\pi G\rho,
\qquad
g=-\nabla\Phi.
$$

For discrete softened source samples,

$$
\Phi(x)=-G\sum_i
\frac{m_i}{\sqrt{\|x-x_i\|^2+\varepsilon_i^2}}.
$$

With fixed softening lengths, differentiating gives

$$
g(x)=G\sum_i
m_i\frac{x_i-x}
{(\|x_i-x\|^2+\varepsilon_i^2)^{3/2}}.
$$

These formulas define a possible numerical model. Point samples are an approximation to extended matter. Softening is not a material's resistance to deformation.

Crucially, `x_i` is a **current physical position**. A deformed or moved region must not continue producing gravity at its original storage coordinates.

Bodies attract through the same mass representation regardless of whether their content came from world generation, building, or a fracture.

### 6.2 Do not select one winning attractor

A player should receive the vector sum of relevant source contributions.

A “dominant source” can be useful for camera preferences, reference-frame selection, or approximation. It should not replace the sum with abrupt switches of physical acceleration.

Between two equal masses, the field may become small through cancellation. That does not mean there is no nearby matter or no tidal effect.

Likewise, a gravitational field sample should not contain only `strength: f32`. A useful query result can include:

~~~rust
struct GravitySample {
    acceleration: DVec3,
    potential: f64,
    tidal: Option<DMat3>,
    error_estimate: GravityError,
    source_epoch: SourceEpoch,
}
~~~

The tidal tensor is the spatial derivative of acceleration. Its computation is optional; a player controller usually does not need it.

Potential values only have meaning under a common boundary condition and additive reference. Do not compare unrelated sector potentials as if their zeros were physically identical.

### 6.3 Start with an adaptive mass hierarchy

A reasonable source summary contains:

- Total mass.
- First moment or center of mass.
- Conservative physical extent.
- Optional second moments.
- Source provenance and revision.
- An error estimate or conservative bound where available.
- Children or an analytic source evaluator.

Far away, an accepted node can replace many sources. Nearby, the query opens that node and requests more detail.

Barnes–Hut-style trees are an established starting point. The familiar near-`O(N log N)` description concerns evaluating forces for many sources under suitable distributions; it is not a worst-case guarantee for every PWC query or edit. [S7]

At first, use the simplest hierarchy that meets measured error and cost targets. Higher multipoles and mutual interactions are extensions, not prerequisites for an initial player-gravity demo.

### 6.4 A chunk is not always a valid point mass

Replacing an entire nearby chunk with one point source can create artificial attraction toward its center. Inside a planet, replacing the whole planet by a point is much worse.

A query inside, overlapping, or close to a source node must descend or use a valid extended-source model.

Possible near-field treatments include smaller mass cells, direct summation over a small source set, or integration of simple volume primitives. All must agree with the selected force kernel.

Do not hide a bad near-field approximation by increasing softening until everything becomes smooth. That also changes real intended behavior.

For the unsoftened inverse-square law, a uniform spherical body's interior field is proportional to distance from its center, and a spherical shell produces zero net field in its cavity. These are valuable analytic reference tests. Softening and a finite-range transition modify that law: compare such implementations with a reference using the same kernel, and measure their deviation from the Newtonian limit separately. [S8]

Exclude a discrete sample's spurious force on itself, but do not exclude an entire assembly's self-gravity when computing its internal loading. Internal gravitational forces contribute to compression even though their net force on an isolated assembly should cancel. A node containing the target must not be accepted as a distant point approximation.

### 6.5 Ungenerated terrain needs a designed mass oracle

A suitable interface might be:

~~~rust
trait ProceduralMatter {
    fn sample_cell(&self, address: CellAddress) -> GeneratedCell;

    fn summarize_mass(
        &self,
        region: ReferenceRegion,
        accuracy: MassAccuracy,
    ) -> MassSummaryResult;
}
~~~

The second function cannot simply be promised for arbitrary procedural code. If the generator is an opaque expensive function, an accurate regional integral may require expensive sampling.

The generator should therefore be designed hierarchically:

1. Large-scale descriptors define major matter distributions.
2. Child descriptors refine them consistently.
3. Fine details add bounded or explicitly approximated corrections.
4. Generated cells inherit those same descriptors.
5. Player edits override the baseline through a separate authoritative record.

A generator may describe a spherical or ellipsoidal concentration as a compact primitive. That is not hardcoded planetary gravity if its mass comes from the same density law and edits can alter it.

For exact hierarchical bookkeeping, parent moments equal the translated sum of child moments. If that cannot hold exactly for the chosen generator, record the discrepancy as approximation error rather than asserting exactness.

**Research risk:** an existing terrain generator may need substantial redesign to provide this contract efficiently.

### 6.6 Baseline plus edits must not count matter twice

Conceptually,

$$
\rho_{\text{world}}
=
\rho_{\text{baseline}}
+
\Delta\rho_{\text{edits}}
+
\rho_{\text{independent assemblies}}.
$$

That expression is valid only if ownership is disjoint and moved baseline material is removed from its old contribution.

When a region is generated in detail, the detailed representation replaces its coarse approximation. It is not added on top of it.

When a mountain detaches and becomes a moving assembly:

1. Its original baseline region becomes masked or otherwise reassigned.
2. Its material receives an assembly owner and pose.
3. The original source contribution is removed.
4. The new source contribution appears at its physical location.
5. Mass and momentum bookkeeping is committed atomically.

A compact procedural subregion plus edits can sometimes represent a huge detached piece without enumerating its voxels. Arbitrary fragmentation can still make that descriptor large. No representation guarantees cheap storage for arbitrarily complex edits.

### 6.7 Signed corrections need special care

Mining naturally creates negative corrections relative to a baseline. A signed correction region can have zero net mass but a nonzero first moment.

Therefore, a generic formula such as

$$
c=\frac{\sum_i m_i x_i}{\sum_i m_i}
$$

is unsafe for a correction tree.

Two reasonable designs are:

- Combine baseline and overrides into a positive physical-mass summary before ordinary tree aggregation.
- Store signed raw moments around fixed expansion origins, with a separate absolute-mass bound for error control.

For an early implementation, positive combined summaries are simpler to reason about. If signed corrections are necessary for efficiency, zero total mass must never be treated as “no gravitational contribution” without considering higher moments.

### 6.8 Local field caches are useful, but are not the authority

Around active players or mechanical regions, a small grid of field samples may reduce repeated tree queries.

Keep three distinct resolutions:

- Source representation resolution.
- Field-query/cache resolution.
- Mechanical deformation resolution.

They answer different error questions.

For instance, gravity around a large uniform mass may be smooth even when its surface collision mesh needs individual block detail. A thin supporting beam may need mechanical refinement although its gravitational contribution is negligible.

If a potential is interpolated and differentiated, use an interpolation with suitable derivative continuity. Trilinear potential interpolation has discontinuous derivatives across cell boundaries. Independently interpolated force vectors may fail to be exactly conservative. Either choice needs a stated error budget.

A normalized field direction is especially unstable near `|g| = 0`. Never normalize without a threshold and a fallback orientation policy.

### 6.9 Update based on error and motion

A moving mass can change gravity quickly without any block edits. Source transforms therefore need an active update path.

A distant, almost unchanged region may remain a compact cached summary. A nearby new mass, a large excavation, or a fast assembly invalidates affected queries promptly.

A periodic full audit can help detect stale summaries. It should not be the primary definition of when gravity changes.

For an unrepresented mass change `|ΔM|` at distances at least `d` from a query, a crude upper estimate is

$$
|\Delta g| \le \frac{G|\Delta M|}{d^2},
$$

provided the geometry and force law justify that distance bound. Cancellation, extended sources, and moved rather than added mass require better estimates.

This gives a principled reason to ignore a tiny distant edit temporarily without ignoring a huge nearby change.

### 6.10 Momentum conservation becomes important for moving worlds

A one-sided approximate tree query is useful for test particles. It does not automatically provide equal-and-opposite interaction forces for two simulated assemblies.

If PWC eventually simulates mutual orbits, collisions, and large moving masses, audit linear and angular momentum. Dehnen's symmetric cell-interaction method is a relevant precedent for linear momentum conservation. It should not be assumed to solve every angular-momentum or timestepping issue automatically. [S20]

GADGET-4 demonstrates that source hierarchy, force approximation, and hierarchical timestepping can be composed into a sophisticated system. Its reported capabilities are evidence for those techniques, not a performance forecast for a voxel game. [S21]

## 7. Materials, amounts, and conservation

### 7.1 Decide what a cell contains

There are three quantities that must not be conflated:

- Material identity or configuration.
- Amount of matter.
- Physical volume.

For a cell containing conserved mass `m` and occupying current physical volume `V`,

$$
\rho=\frac{m}{V}.
$$

If it stretches without exchanging matter, its mass stays fixed and its density changes. If an algorithm recalculates `m = constant_density × new_volume` after every deformation, it creates or deletes mass.

That is a bookkeeping error unless the model explicitly includes a material source or sink.

### 7.2 The reaction specification leaves a real choice

The current candidate reaction function preserves element occurrences while their resource coordinates can change.

Suppose mass were assigned as `m(element_identity)`. A reaction that changes identity would generally change total mass even while preserving occurrence count.

There are three coherent responses:

| Policy | Benefit | Cost |
|---|---|---|
| Every conserved constituent amount carries the same reference mass | Compatible with constituent-count conservation | Different densities must arise through packing, amount per volume, or reference-volume differences |
| Constituent amounts carry conserved masses independent of mutable resource coordinates | Supports more distinctions | Requires additional state and a well-defined transfer rule |
| Reactions may change mass under a separate accounting law | Very flexible fictional physics | Requires explicit energy/source rules and changes the current reaction contract |

**Recommendation:** start with a conserved matter amount whose gravitational and inertial mass agree. Do not derive mass directly from the mutable coordinates of the current reaction candidate.

This does not prohibit dense and light materials. Their equilibrium volume, porosity, amount per occupied cell, or packing behavior can differ. Those differences need a rule connecting the material configuration to mechanical state.

Adding a continuous amount multiplier outside the existing configuration is itself an extension. Its interaction with reaction rates, constituent accounting, transfer, and the candidate's finite configuration capacity must be defined. The current two-configuration function cannot be assumed to handle arbitrary split or merged amounts unchanged.

### 7.3 “One block” becomes an ambiguous inventory unit

Curved or adaptive cells can have different physical volumes. The following three statements cannot all remain true:

1. Every full cell contains the same mass.
2. The same material has the same density everywhere.
3. Cells have unequal volumes.

The most physically coherent target is to trade **amounts of material**, with a block-sized placement being an operation that fills a cell of a particular reference volume.

A familiar inventory UI can display nominal block equivalents. Internally, conservation uses amounts.

If PWC instead prefers one inventory item per cell regardless of shape or size, that is a valid gameplay simplification. It should be documented as such because shell transitions and remeshing will otherwise create duplication exploits or inexplicable density changes.

This choice is more fundamental than the exact gravity solver.

### 7.4 Stiffness, strength, and density are different properties

| Quantity | Meaning | Why it matters |
|---|---|---|
| Mass/amount | How much matter is present | Gravity and inertia |
| Reference density or volume | Preferred packing at a specified state | Initial geometry and pressure response |
| Shear stiffness | Resistance to small shape change | Whether a structure bends visibly before yielding |
| Bulk stiffness | Resistance to volume change | Compression and density |
| Yield threshold | Onset of irreversible deformation | Whether a large shape can persist |
| Flow timescale or viscosity | Speed of sustained deformation | Whether relaxation takes seconds or geological time |
| Cohesion/fracture resistance | Resistance to separation | Cracks and fragmentation |
| Friction | Resistance to tangential sliding at contact | Rubble, slopes, and assembled structures |

An initial elastic prototype needs fewer properties than the complete system. It must nevertheless distinguish stiffness from yield strength.

A material can be stiff but brittle. Another can deform elastically a great deal without permanently changing shape. One scalar “resists warp” cannot express both.

### 7.5 How to connect these properties to the emergent material system

A suitable bridge is a versioned function:

~~~rust
fn mechanical_response(
    configuration: &MaterialConfiguration,
    state: &MechanicalMaterialState,
) -> ConstitutiveParameters;
~~~

This function may eventually derive its parameters from your universal configuration law. It does not need named stone/metal exceptions.

However, the existing reaction score is not automatically mechanical energy. It has no established dependence on spatial strain. Differentiating it with respect to resource coordinates does not produce a physical elastic modulus.

A legitimate derivation requires a model that says how spatial deformation changes material energy or response. Until that bridge exists, use clearly labelled prototype parameter assignments to test the engine.

Do not disguise those assignments as a discovered consequence of the current reaction mathematics.

### 7.6 Shared material boundaries must fit

Two neighboring materials cannot independently move a shared face by different amounts while still occupying a continuous solid.

For bonded material, they share boundary positions and transmit traction. Different constitutive properties lead to different internal strain and load distribution through a coupled solve.

For unbonded material, separate surfaces may slide or separate under contact rules.

Multiplying each block's vertex warp by its own resistance value is not a substitute. It introduces cracks or overlap unless the shared geometry is reconciled.

A super-strong inclusion can resist bending, but it may sink, pull neighboring material, or tear its attachments. Strong does not mean globally immovable.

A huge cube that resists rounding will also retain surfaces that are not everywhere perpendicular to its gravitational field. Its faces can feel like slopes. Preserving that shape and forcing every face to be gravitationally horizontal are conflicting goals; material strength cannot remove the conflict.

### 7.7 Hydrostatic pressure remains part of the model

Write stress as

$$
\sigma=-pI+s,
\qquad
p=-\frac13\operatorname{tr}\sigma,
\qquad
\operatorname{tr}s=0.
$$

A simple metal-like yielding model responds primarily to the deviatoric part `s`. For example,

$$
\sigma_{\rm eq}=\sqrt{\frac32\,s:s}
$$

can be compared with a yield threshold.

That is a useful prototype, not a universal model of soil, rock, and loose material. Frictional granular materials can have pressure-dependent strength and compaction. [S12], [S13]

For PWC, a useful simplification is strong resistance to volume change with slower shape relaxation. Keep the pressure needed to balance gravity even if visible compression is small.

Deleting pressure from the equilibrium equations would remove the mechanism supporting a planet's interior. And a cavity cut into a compressed solid changes the local loading; “mostly hydrostatic” does not guarantee that every deep tunnel remains stable.

## 8. Mechanics: how matter should actually move

### 8.1 Begin with the balance law

In current physical coordinates,

$$
\rho\frac{Dv}{Dt}
=
\nabla\cdot\sigma+\rho g+f_{\rm other}.
$$

In words: acceleration comes from internal forces, gravity, and other applied forces.

A slowly changing equilibrium approximately satisfies

$$
\nabla\cdot\sigma+\rho g+f_{\rm other}=0,
$$

with appropriate boundary and contact conditions.

This formulation includes free fall, self-gravitating compression, a supported beam, and tidal loading. It also explains why a gravity sample alone does not determine stress: the stress depends on the surrounding structure and its boundary conditions. [S10]

### 8.2 Separate rigid motion from internal deformation

An assembly may translate or rotate substantially while barely changing shape.

Represent its bulk pose separately from deformation coordinates. This improves numerical conditioning and prevents a stiffness model from resisting ordinary rotation.

In a freely falling nearly uniform external field, the assembly's center of mass accelerates. Only the residual loading after accounting for bulk acceleration and rotation should contribute to its internal deformation.

If mechanics is solved in an accelerating or rotating frame, the corresponding inertial terms must be included. Simply subtracting a center position and continuing to use an inertial equation is not generally correct.

For the first dynamic implementation, integrate physical motion in local non-rotating frames and use assembly coordinates chiefly for storage and geometry. Introduce rotating-frame dynamics only when its equations are explicit.

### 8.3 A gravitational stress scale is a diagnostic

For a self-gravitating object with density `ρ` and characteristic size `L`,

$$
\sigma_g\sim G\rho^2L^2.
$$

Comparing that scale with strength `Y` gives

$$
\Pi_g=\frac{G\rho^2L^2}{Y}.
$$

Small `Π_g` suggests strength-dominated behavior; large `Π_g` suggests that self-gravity may exceed available strength.

This explains why a single ordinary block can remain effectively undeformed while an enormous aggregate of the same material may relax.

But `L` is a structural scale, not a value available from a single gravity sample. Shape, supports, rotation, and constitutive response matter. The ratio is useful for initialization, scheduling, and test selection. It should not directly set a per-voxel displacement.

For an illustrative `ρ = 3000 kg/m³` and `Y = 1 MPa`, the crude equality gives `L ≈ 40.8 km`. This is a dimensional estimate, not a predicted universal roundness threshold for real asteroids.

### 8.4 A concrete first deformable solver

**Engineering recommendation:** use an adaptive coarse mechanical mesh over the occupied material, with finer resolution around important interfaces, cavities, slender structures, and active edits.

Tetrahedral elements are a reasonable first internal discretization because their local volume and affine deformation are straightforward to inspect. The visible and editable cells can remain hexahedral patches; the mechanics mesh need not be the gameplay voxel grid.

The correspondence between the two meshes must be explicit. A tetrahedron spanning an empty cave is not allowed to transmit solid stress merely because its bounding box contains rock.

For each mechanical element, retain:

- Reference node positions and rest volume.
- Current node positions and velocities.
- Conserved mass distribution.
- Material parameters or a justified homogenized response.
- Optional plastic state and damage.
- Links to affected gameplay cells and material boundaries.

Coarsening material is not just averaging density. Mechanical connectivity, directional stiffness, and voids can matter far more than mass.

### 8.5 A starting elastic energy

A rotation-invariant compressible neo-Hookean energy is one possible reference model:

$$
W(F)
=
\frac{\mu}{2}
\left(\operatorname{tr}(F^{T}F)-3\right)
-\mu\ln J
+\frac{\lambda}{2}(\ln J)^2,
\qquad J=\det F>0.
$$

Here `μ` and `λ` are elastic parameters. This is a constitutive choice, not the uniquely correct fictional material law.

A standard hyperelastic construction derives stress from an energy depending on deformation. Frame indifference ensures that a rigid rotation does not create elastic energy. [S11]

For a simplified implicit dynamic step, consider

$$
\mathcal{E}(x)
=
\frac{1}{2\Delta t^2}
\sum_i m_i\|x_i-\widehat{x}_i\|^2
+
\sum_e V_{0,e}W(F_e)
+
E_{\rm grav}(x)
+
E_{\rm contact}(x).
$$

The predicted positions `x̂` incorporate previous velocity and any forces **not already included** in the energy. In this expression gravity belongs to `E_grav`; it must not also be applied in the predictor.

For self-gravity, the potential energy has the familiar pairwise one-half accounting. When gravity is approximated as a frozen external field during a substep, use the corresponding external-force treatment instead of mixing inconsistent definitions.

This is a useful mathematical baseline for comparison. It is not a complete production solver: contact, constraints, conditioning, plasticity, and convergence policy still have to be implemented.

### 8.6 Near incompressibility has numerical costs

A high bulk stiffness suppresses volume change. It can also make the solve poorly conditioned.

Some low-order element choices become artificially stiff under near-incompressibility. Adding a very large numerical constant is not a robust solution.

Evaluate a suitable mixed formulation, a validated volumetric constraint method, or a deliberately limited compressibility. The choice should be judged through compression and bending tests rather than by appearance alone.

### 8.7 XPBD is a useful experiment, not an automatic answer

XPBD provides a compliant-constraint formulation that improves the interpretation of stiffness relative to traditional position-based dynamics. It is a plausible prototype for coarse elastic constraints.

Its own paper also discusses residual artificial compliance when iterations stop before convergence. Thus “XPBD” does not mean that ten iterations accurately transmit load across a planet, nor that fracture stresses are already trustworthy. [S14]

Compare it with a small reference FEM solve on simple cases. If the iterative method produces different physical behavior when the mechanical mesh is refined or the iteration count changes, that error must be measured.

### 8.8 Irreversible relaxation requires a history law

Pure elasticity tries to return material to its reference shape when loads are removed. Long-term rounding of weak matter generally needs plasticity, creep, flow, or fracture.

A common finite-deformation organization is

$$
F=F_eF_p,
$$

where `F_e` is elastic deformation and `F_p` records irreversible change.

For a first restricted model, choose one published constitutive formulation and implement its return mapping, hardening or viscosity, and state transfer coherently. Do not combine an arbitrary yield test with an unrelated update to `F_p` and call the result physically established.

The earlier two-property model of density plus strength does not determine:

- How much material deforms per second.
- Whether deformation changes volume.
- Whether unloading reverses it.
- Whether it flows or cracks.
- How behavior changes under compression.

Those are separate choices even in a stylized universe.

### 8.9 The coupled update loop

A candidate server workflow is:

~~~text
1. Read an immutable snapshot of matter, pose, geometry, and bonds.
2. Update physical source summaries for that snapshot.
3. Estimate force and mechanical approximation errors.
4. Select active mechanical regions and their boundary coupling.
5. Evaluate gravity and other loads.
6. Solve one bounded mechanical increment.
7. Update constitutive history and propose any bond failures.
8. Construct candidate physical geometry.
9. Validate cells, shared boundaries, swept contacts, and conservation.
10. Refine or reduce the step if validation fails.
11. Commit the accepted geometry and topology transaction.
12. Refresh mass summaries and publish the accepted state.
~~~

For substantial deformation, gravity must be updated as matter moves. A single solve using the original source distribution can become inconsistent.

For small changes, a staggered update may be sufficiently accurate. Validate that approximation against smaller timesteps and tighter coupling on selected benchmarks.

### 8.10 Quasi-static relaxation needs a domain of validity

Updating a geological equilibrium every few hours can be sensible when changes are slow and inertia is negligible.

It is inappropriate for an impact, a collapsing support, a rapidly rotating fragment, or a moving mass passing close to a player.

Use different timescales for different jobs:

- Active motion and contact: ordinary physics ticks.
- Changing local gravity: sufficiently frequent source/query updates.
- Slowly evolving internal strain: controlled mechanical substeps.
- Expensive remeshing or equilibrium proposals: background jobs.
- Audits and compaction: periodic maintenance.

A wall-clock timer does not by itself define a physically meaningful relaxation rate. The chosen material timescale should.

### 8.11 Damping is a declared approximation

If a planet loses gravitational potential energy while settling, that energy goes somewhere: kinetic energy, heat, damage, or a modelled dissipative sink.

PWC need not initially simulate temperature. It should still distinguish deliberate dissipation from numerical energy creation.

Record diagnostic energy changes in experiments. Avoid promises of energy conservation for a heavily damped relaxation model.

Also avoid center-of-mass drift from damping internal motion incorrectly. Internal relaxation should not arbitrarily brake the entire assembly's orbit.

## 9. Geometric validity is a system contract

### 9.1 Separate local validity, shape quality, and global collision

These are different requirements:

| Requirement | What it detects | What it does not establish |
|---|---|---|
| Positive local Jacobian | Local orientation and non-collapse | Global absence of overlap |
| Bounded singular values | Excessive stretch, compression, or anisotropy | Correct material physics |
| Shared-boundary agreement | Cracks between bonded neighboring cells | Absence of remote self-contact |
| Self/inter-assembly collision checks | Interpenetration | Good cell shape |
| Conservation checks | Amount/accounting errors | Valid geometry |
| Bounded surface approximation | Rendering/collision mismatch | Correct long-term dynamics |

No one scalar replaces the table.

### 9.2 Measure both strain and visible cell shape

For mechanics, compare current geometry with the reference geometry through `F`.

For building quality, inspect the actual physical cell geometry. A reference cell can already be badly skewed while `F = I`.

Useful quality measurements include:

- Edge-length ratio.
- Face-angle deviation.
- Volume relative to nominal cell volume.
- Jacobian singular values after normalizing coordinate units.
- Distance between neighboring surfaces that should coincide.
- Curvature or interpolation error across a rendered face.
- Deviation of intended “up” faces from local gravitational horizontal.

Suggested prototype reporting thresholds, not established acceptable values:

- Report ordinary construction cells exceeding a 1.05 longest/shortest-edge ratio.
- Report face angles deviating more than 3° from a right angle.
- Highlight all exceptional topology and all cells beyond looser emergency bounds.
- Measure the fraction and spatial distribution of failures, not only the average.

A standard six-face cube-sphere may fail these proposed thresholds over substantial regions. The point is to discover that immediately.

Do not force the solver to satisfy arbitrary shape limits by silently making affected material infinitely strong. A representation-quality failure should trigger refinement, a supported retiling operation, or a visible failure of the proposed model.

### 9.3 A determinant check at a few points is insufficient for curved cells

For a piecewise-affine tetrahedron, the deformation gradient is constant within the element, so one correctly computed signed volume characterizes local orientation.

For a trilinear hexahedron, the Jacobian varies internally. Positive values at its corners or center are not a general proof of validity. Research specifically addresses robust hexahedron validation using polynomial bounds and subdivision. [S15]

Therefore, choose an authoritative geometry representation and validate **that representation**:

- Piecewise-affine tetrahedra with a consistent shared-face subdivision.
- Trilinear cells with a suitable certified or conservative validity method.
- Analytic mappings with proven bounds over their supported domains.
- Higher-order cells with corresponding derivative bounds.

Checking one auxiliary mesh does not automatically validate a different curved render mapping.

### 9.4 Local positivity is not global injectivity

Two distant parts of the same object can pass through each other while each small element retains a positive determinant.

You need collision handling between non-adjacent occupied surfaces, and between assemblies, in addition to local element checks.

The geometry of empty cells should not become an invisible solid obstacle. Collision follows occupied material and relevant physical surfaces.

### 9.5 Validate motion, not just endpoints

For each candidate update:

1. Bound displacement and strain change.
2. Detect swept surface contacts.
3. Limit the step before collision or element collapse.
4. Re-evaluate quality over the accepted increment.
5. Commit only a feasible state.

Interpolating validated endpoint snapshots on a client is still a trajectory choice. A render interpolation must not show material passing through a player while the server uses another path.

IPC is a useful research reference because it combines barrier energies with a solver designed to preserve collision and inversion feasibility. Its guarantees depend on its discretization, collision handling, and solver assumptions. Attaching the name IPC to a generic warp does not inherit those guarantees. [S16]

### 9.6 Use shared geometry at bonded interfaces

Adjacent patches should agree on their common physical boundary through shared control data or enforced interface constraints.

Do not independently evaluate two supposedly equivalent mappings with unrelated approximations and hope their vertices coincide.

At an adaptive interface, one coarse face may meet several fine faces. Define:

- Which side owns the canonical boundary description.
- How the boundary is subdivided.
- How neighbor traversal identifies the correct subface.
- How collision and rendering share the subdivision.
- How fluxes, forces, and material transfer are accumulated.

A graphics skirt can conceal a crack. It does not establish a valid collision or material interface.

### 9.7 A failed solve needs a bounded, explicit outcome

If a candidate cannot be validated, preserve the last valid committed state.

Then reduce the step, refine the relevant region, or use a supported lower-complexity representation.

This is a numerical containment policy, not a proof that the physical simulation is complete. Repeatedly freezing weak material because it cannot be represented would falsify the architecture's promised relaxation capability.

Track those failures as first-class diagnostics. A production feature should not depend on them being rare by assumption.

## 10. Constructing usable voxel layouts

### 10.1 Treat gravity alignment as a meshing preference

Where `g ≠ 0`, a useful vertical preference is

$$
\hat u=-\frac{g}{\|g\|}.
$$

For a conservative non-rotating gravitational field, equipotential surfaces are perpendicular to gravity. That makes them useful candidates for approximately horizontal layers.

But `û` supplies only one direction. It does not choose the other two axes, their scale, their twist, their grid-line correspondence, or the treatment of field zeros.

A collection of independently rotated cubic frames generally fails to integrate into a coherent coordinate map. Compatibility imposes additional conditions.

Use gravity as one input to layout construction alongside:

- Surface and boundary geometry.
- Existing construction alignment.
- Cell-size and distortion targets.
- Patch interfaces.
- Places where defects or changes of resolution are allowed.
- The cost of preserving existing addresses.

This is a meshing policy. It is not an additional force acting on material.

### 10.2 World generation may choose a convenient initial layout

A generator that creates a nearly spherical mass can also construct an appropriate patch atlas for it.

This is compatible with emergence. The generated mass still has ordinary material properties and matter-derived gravity. The runtime can later discover a different mass distribution after edits.

The crucial restriction is that subsequent gravity changes do not silently replace the atlas. The atlas is part of the world state.

Generated geometry should be a direct initial condition. There is no need to simulate billions of years of collapse merely to justify the starting world.

If that initial condition is intended to be mechanically equilibrated, initialize its prestress or equilibrium state accordingly. Otherwise, enabling mechanics may cause an unintended global settling event.

### 10.3 Prototype a cube-sphere honestly

A six-chart shell is an excellent first test because its limitations are understandable.

Test at least:

- The simple normalized projection as a deliberately imperfect baseline.
- An equiangular or other documented improved mapping.
- Face edges and all eight corners.
- Multiple depths and altitudes.
- Buildings that cross a chart boundary.
- Identical blueprints placed in different regions.

Bowerbyte's Blocky Planet is particularly relevant: its development account discusses projection distortion, shell transitions, and multi-resolution neighbor relationships. It also reports a then-unresolved hollow-core limitation caused by its chunk scheme. This is evidence for a working spherical voxel prototype, not evidence that all the deep-interior problems are already solved. [S17]

The starting planet's large radius helps normal variation and ordinary-depth distortion. It does not make the prototype exempt from tests at awkward locations.

### 10.4 Do not run the same angular columns into the origin

With a fixed angular layout,

$$
\ell_{\rm horizontal}(r)
\propto r.
$$

If a horizontal cell dimension is `ℓ₀` at radius `R`, it becomes approximately

$$
\ell(r)=\ell_0\frac{r}{R}.
$$

At half-radius it is half as wide. Near the origin it collapses.

There are three real choices:

| Interior strategy | Benefit | Cost |
|---|---|---|
| Limit the curved layout to a shallow shell | Simplest and excellent over ordinary depths | Does not fulfill unrestricted core access by itself |
| Reduce angular resolution through depth | Keeps sizes within a bounded range | Introduces refinement interfaces and irregular adjacency |
| Use a multi-block interior with a finite central region | Avoids a radial singularity | Requires a designed transition layout and distortion analysis |

Because you explicitly want players to visit and build deep inside, the final design should include the latter two possibilities. “Nobody will reach the center” is not an acceptable final assumption.

### 10.5 A concrete interior prototype

Use a finite central Cartesian region and a surrounding set of mapped blocks connecting it to an outer spherical shell.

This family of layout can be constructed with ordinary volume-meshing techniques. It can eliminate collapse at the center, but does not by itself guarantee nearly cubic cells everywhere.

For the first prototype:

1. Build a small explicit central region.
2. Surround it with a handful of transition blocks.
3. Match shared boundary vertices exactly.
4. Subdivide each block into structured cells.
5. Measure cell shape through the whole volume.
6. Identify where additional blocks or refinement transitions reduce poor quality.

Do not immediately extrapolate a successful small mesh to a planet with trillions of addressable cells. The representation must be procedurally describable, and singular regions must remain manageable at all scales.

The central Cartesian region is a coordinate choice, not a stronger special material.

### 10.6 Depth transitions change cell relationships

Simply doubling angular resolution produces one-to-four relationships across a radial face. That is more than a rotated face index.

The neighbor API must return a set of face connections with subregions, orientations, and area information where needed.

For ordinary equal-resolution neighbors, the representation should collapse to a fast direct lookup. Exceptional interfaces should pay their own extra cost.

Numerical mechanical refinement must also be distinguished from gameplay-cell refinement. Refining an internal solver mesh need not create inventory items or change editable block addresses.

### 10.7 High altitude has the inverse problem

Fixed angular columns get wider with increasing radius. Extending a planet's curved lattice across all empty space eventually gives absurdly large building cells and conflicts with neighboring worlds.

Treat curved terrain patches as spatially bounded layouts. Empty universe space need not belong to any planetary chart.

A structure extending outward can retain its attached local layout until it reaches a quality or extent limit. Further construction may create another compatible patch or an independent assembly with explicit joints.

The boundary must be a documented building operation. An object should not secretly stretch because it crossed an arbitrary chart's preferred altitude.

### 10.8 Arbitrary new worlds need more than a sphere detector

For a general mass distribution, a future automatic layout pipeline could:

1. Inspect occupied physical geometry and existing material organization.
2. Sample gravity where it is well conditioned.
3. Construct a preferred frame field with cubic symmetry.
4. Repair or choose admissible singularity structure.
5. Build a compatible volumetric parameterization.
6. Quantize it into usable cell counts.
7. Extract and validate the resulting cells.
8. Preserve or transfer material and attachment state.

The literature has made substantial progress on individual stages. Locally Meshable Frame Fields addresses topological problems in direction fields; Integer-Sheet-Pump Quantization addresses integer layout choices; later work permits more flexible singularities. None of those titles means that the complete persistent multiplayer PWC problem is solved. [S6], [S22], [S23]

A 2026 neural frame-field preprint is another relevant research direction, but a continuous predicted field still needs a valid discrete layout and state-transfer pipeline. I would not make a neural model a dependency of authoritative world geometry. [S24]

### 10.9 Keep existing buildings out of unnecessary retiling

A patch can remain physically attached to a structure while other material around it changes representation.

This should follow meaningful mechanical and material distinctions: persistent solid structure, attachments, or low deformation. It should not require an unexplained “player-built blocks are immune” exception.

There is a real tradeoff: preserving an entire existing structure may force a less ideal surrounding grid. The layout objective should prefer preserving useful state over cosmetically perfect gravity alignment.

## 11. Fracture, splitting, merging, and retiling

### 11.1 A split is a material event

A planet should not split because a gravity classifier changed the number of detected wells.

It splits because material continuity or bonds are removed, fail, or allow separation. Gravity then influences the pieces' subsequent motion.

A hollow shell may be one connected structure despite weak or zero interior gravity. Two unconnected bodies can share one gravitational potential well. These examples are enough to show why gravity domains cannot define mechanical ownership.

### 11.2 The sequence for a fracture

A generic fracture transaction should:

1. Identify the material connections that fail.
2. Create separate boundary descriptions where a shared boundary becomes a crack.
3. Remove the corresponding force-transfer constraints.
4. Determine independently moving components or update their connectivity representation.
5. Preserve each component's current physical geometry.
6. Transfer velocity consistently.
7. Rebuild affected collision and mass summaries.
8. Publish the event and update attachments.

A fragment initially keeps its inherited shape. It does not immediately become a sphere or reset its voxel rows.

Connected-component discovery can be hierarchical and incremental. An arbitrary edit can still cut a large load path, so the worst case cannot be dismissed as a constant-time chunk update.

### 11.3 Preserve velocity, not only position

For a piece released from a rotating assembly, its initial velocity contains the old assembly's translational motion plus rotational motion at that location.

If that contribution is lost, detached fragments will appear to stop or receive artificial impulses.

When splitting a deforming body, preserve the local velocity field as well. Compute fragment center-of-mass and angular momentum from the transferred state, rather than inventing a new stationary reference frame.

Rebasing coordinates should not change physical position or velocity.

### 11.4 A half-planet need not remain a half-sphere

Your idea that a fragment might be “half or a third of a sphere” is correct as a description of an immediate fragment or a supported long-lived shape.

It is not necessarily an equilibrium under self-gravity. A sufficiently weak isolated half-sphere can continue rounding. A strong one can preserve a large flat fracture surface. Rotation and nearby masses can alter the outcome.

The solver should produce the behavior implied by the model. It should not preserve every fragment shape just because the fragment was once part of a sphere.

### 11.5 Contact is not automatically merging

Two worlds or assemblies that touch may bounce, slide, crush, or weld.

They should not be forced into one coordinate grid merely because their surfaces meet or their gravitational fields overlap.

A weld can add mechanical constraints between existing patches. It does not require all their storage coordinates to become contiguous.

This is particularly useful for a space station assembled from independently constructed sections. Its components may retain different local grids while sharing mechanical connections.

### 11.6 Large flow eventually needs a representation change

If matter flows extensively, keeping its original hexahedral cells forever may produce unusable shapes even after plastic stress has relaxed.

There are three possible responses:

- Preserve the material cells and tolerate substantial distortion.
- Replace their layout through conservative retiling.
- Transfer highly deforming matter to another representation, such as particles or an Eulerian material field.

The first best preserves individual block identity. The latter two better support unrestricted flow.

**Recommendation:** preserve ordinary solid structures; allow explicit reorganization for material that the physical model treats as loose, flowing, or extensively rearranging. Base that distinction on state and response, not named material categories.

MPM is relevant here because its particles and background grid handle substantial deformation differently from a permanent mesh. It does not automatically retain exact editable block topology or every form of fracture behavior. [S18]

### 11.7 A conservative transfer model

Suppose old cells `i` contain material masses `m_i`, and new cells `j` cover the same occupied physical region.

Define nonnegative transfer weights `w_ji` with

$$
\sum_jw_{ji}=1
$$

for every old cell. Then transfer each material amount by

$$
m'_j=\sum_iw_{ji}m_i.
$$

This conserves mass algebraically.

That is only the beginning. The transfer must also account for:

- Material composition without silently mixing distinct immutable configurations.
- Linear momentum.
- Angular momentum to the selected tolerance.
- Energy and declared numerical dissipation.
- Plastic state and damage.
- Machine entities, inventories, and attachments.
- Broken versus intact boundaries.
- Exact occupancy near thin structures.

Intersection-volume weights are one possible construction. They assume a density model within old cells and can be expensive. Sampling introduces approximation and must not silently erase a thin wall.

For momentum, transferring mass and velocity independently is generally insufficient. Transfer momentum amounts and derive velocity from them. Angular momentum may require additional moments or a corrective solve.

### 11.8 Regridding is not permission to move buildings

A coordinate-only operation should preserve occupied physical geometry within a declared tolerance.

If a proposed retiling changes the outer shape, it has included physical or geometric modification. That change must be accounted for, not hidden under “new coordinates.”

Likewise, resetting the reference shape without transferring stress or plastic history can erase stored elastic energy. A convenient rest-state reset is a physical change unless the model explicitly justifies it.

This is one of the highest-risk implementation areas. A good first version should avoid routine remeshing of active player structures.

### 11.9 Stable references after retiling

Maintain a bounded translation record from old handles to replacement regions or new handles.

For important entities, reattach through physical material support or preserved attachment coordinates. A one-to-many cell replacement cannot always be represented by one new integer address.

Queries using stale handles should produce a defined result: resolve, report replacement, or fail cleanly. They should never edit an unrelated cell that inherited the same numeric index.

## 12. Movement, building, collision, and rendering

### 12.1 Local movement should use physical distances

A logical step of one in a distorted patch does not necessarily equal one meter.

If `J = ∂x/∂q`, the coordinate metric is

$$
H=J^TJ,
\qquad
ds^2=dq^TH\,dq.
$$

This explains why blindly using ordinary Euclidean movement in patch coordinates can change walking speed by location.

**Recommendation:** perform nearby entity motion and collision in a physical local frame. Use patch coordinates for addressing and material attachment. Avoid requiring the entire movement engine to solve curved-coordinate equations from the start.

### 12.2 Gravity orientation and camera orientation are different

The physical acceleration should remain the computed vector.

The player controller can construct an up direction from `−g` where it is sufficiently strong. The camera may approach that direction at a limited angular rate for comfort.

Near zero gravity:

- Keep a meaningful prior orientation or use the current support/assembly frame.
- Do not divide by an almost-zero magnitude.
- Allow intentional orientation control.
- Do not invent gravitational acceleration merely to preserve an upright camera.

A support normal can differ from gravitational up on a slope. The controller still needs a slope, contact, and step policy.

### 12.3 Rotation complicates “horizontal”

On a steadily rotating world, a stationary observer in the rotating frame experiences an effective acceleration

$$
g_{\rm eff}
=
g-\Omega\times(\Omega\times r).
$$

Motion in that rotating frame also introduces Coriolis terms, and changing rotation introduces additional terms.

Therefore, gravitational equipotentials alone do not describe every effective horizontal surface of a rotating world.

An initial non-rotating world is a legitimate simplification. If rotation is added, make the reference-frame equations part of that feature rather than assuming the original gravity-alignment rule remains exact.

### 12.4 Moving surfaces carry velocity

Using

$$
x=c+R\chi(q,t),
$$

the physical velocity of an entity moving through local coordinates includes

$$
v=
\dot c
+\omega\times(R\chi)
+R\left(
\frac{\partial\chi}{\partial t}
+
D\chi\,\dot q
\right).
$$

For material attached at fixed `q`, the last coordinate-motion term vanishes. For a walking player, it generally does not.

This matters for jumping from a rotating fragment, riding a deforming platform, friction, and boarding a ship. Updating only the rendered surface position while ignoring its velocity gives incorrect contact behavior.

### 12.5 Raycasting cannot blindly reuse straight-grid DDA

A straight ray in physical space generally becomes a curve under the inverse of a nonlinear patch mapping.

Transforming only the ray origin and direction and then running ordinary straight-line grid DDA is therefore not exact.

A practical route is:

1. Intersect conservative physical bounds of nearby patches/chunks.
2. Traverse candidate occupied cells or a surface acceleration structure.
3. Intersect the authoritative physical cell surfaces.
4. Return the cell handle, face/subface, hit point, and geometry epoch.

A local affine approximation can accelerate this when its error is bounded. It should be an optimization with a fallback.

Do not assume a warped chunk is bounded by its transformed corners alone. Nonlinear mappings can have extrema between them. Use analytic bounds, conservative subdivision, or bounded displacement inflation.

### 12.6 Collision geometry must agree with visible geometry

Choose one cell-surface definition:

- A specified triangulation.
- Bilinear or curved faces with a supported collision method.
- A certified approximation with a declared tolerance.

Adjacent cells must share the same boundary interpretation. Two different diagonals across a nonplanar quad can produce different surfaces.

Use simplified collision only where its error is small relative to the player's scale and the relevant movement. A visually curved wall with an unrelated unwarped collision plane will quickly destroy trust in the building system.

### 12.7 Greedy meshing remains useful, with a curvature limit

Merging many coplanar logical faces does not imply that their mapped physical surface is planar.

For a spherical arc approximated by a chord of length `L`, the maximum deviation is approximately

$$
e\approx\frac{L^2}{8R}.
$$

At `R = 10^7` meters and `L = 1000` meters, that is about 1.25 centimeters. Tiny angular change does not imply zero positional error.

A merged face can therefore be subdivided until both world-space and screen-space error targets are met.

A vertex shader only transforms existing vertices. It does not automatically curve the interior of a giant two-triangle quad.

Normals also need the correct transform. For a smooth invertible local map, a normal transforms proportionally to the inverse transpose of the Jacobian, followed by normalization. A rigid rotation is sufficient only for a rigid map.

### 12.8 Keep graphics LOD separate from simulation state

Changing render distance, screen resolution, or camera location must not alter:

- Material amount.
- Cell addressability.
- Gravity.
- Mechanical support.
- Reaction opportunities.
- Authoritative collisions.

Distant visual LOD can approximate the same authoritative geometry. A close player must recover consistent block-level surfaces without a physical world change.

Geometry epochs should invalidate or update bounding volumes, occlusion data, shadow geometry, and mesh caches together as needed.

### 12.9 Use hierarchical coordinates for enormous worlds

A global `DVec3` is helpful but not an infinite-precision coordinate system.

A practical model is:

~~~rust
struct UniversePosition {
    sector: I64Vec3,
    offset: DVec3,
}
~~~

The sector size is fixed by the format; offsets remain within a controlled range.

Physics works relative to nearby origins. GPU vertices are camera- or patch-relative floats. Subtract large positions in a sufficiently precise representation **before** converting to `f32`.

At magnitude `10^7`, ordinary `f32` spacing is about one meter. A planet of that radius makes absolute single-precision positions unsuitable for block-scale movement. Large-world engine documentation describes the same underlying precision problem. [S25]

Do not send already-rounded huge float coordinates to a shader and expect camera subtraction to recover their lost bits.

### 12.10 Building across boundaries needs explicit semantics

Ordinary placement extends the targeted patch when that produces valid geometry and a well-defined neighboring cell.

At an exceptional interface, use the actual face/subface relation. If there is no suitable existing cell layout, placement may create a new patch or assembly under the normal construction rules.

Blueprints should declare whether they preserve:

- Local cell connectivity.
- Physical meter dimensions.
- Attachment to an existing surface.
- A rigid internal geometry.

These are not always equivalent on curved terrain.

A rigid factory floor may span local terrain with foundations. A large terrain-following wall may intentionally curve. The engine should make that distinction available rather than silently choosing based on a planet label.

## 13. Multiplayer, persistence, and authoritative commits

### 13.1 The server owns physical geometry

Clients can predict movement and ordinary edits. The authoritative server decides:

- Material contents and amounts.
- Current assembly poses and deformation.
- Mechanical connectivity and fracture.
- Committed layout changes.
- The source state used for gravity.
- The simulation-law and generator versions.

A background solver produces a proposal. It does not mutate the live world halfway through its solve.

This fits the current server-oriented direction, but extends the state that the server must retain.

### 13.2 Use several revisions, not one overloaded number

Useful revision domains include:

| Revision | Meaning |
|---|---|
| Cell/content revision | Material or amount changed |
| Patch topology revision | Cell layout or adjacency changed |
| Geometry epoch | Physical embedding changed |
| Assembly pose tick | Bulk pose and velocity state |
| Source epoch | Gravity source snapshot changed |
| Law fingerprint | Rules, numerical conventions, and content versions |

A transaction records the revisions it was computed from.

These numbers need not all advance together. A rigid translation can change pose and source placement without changing local material contents. A reaction can change material response without changing topology.

The commit layer must still enforce a coherent set of dependencies.

### 13.3 Validate an edit in physical space

An edit request should identify its target handle, expected content/topology revision, and the physical context needed for reach validation.

The server resolves the target against an appropriate authoritative state. If geometry has moved significantly since the client's view, the request needs a defined rejection or revalidation path.

Do not validate reach in old unwarped coordinates while applying the edit to current curved geometry.

A clean rule for simultaneous edits is still possible: one accepted revision wins, and conflicting predictions are corrected.

### 13.4 Commit a geometry change atomically

A deformation/topology commit may include:

- New control points or a compact deformation descriptor.
- Changed connections.
- Updated ownership and handles.
- Conservative mass/momentum transfer records.
- Updated physical bounds.
- An activation tick.
- A trajectory or motion descriptor for interpolation.
- Required cache invalidations.

Clients receive the change before its activation when practical. A late client receives a suitable snapshot and correction.

Clients should not independently optimize the world and hope their nonlinear solvers make identical decisions. They can evaluate compact committed mappings and predict limited motion.

### 13.5 Revalidate background results

While a deformation proposal is being calculated, players may mine, build, or trigger other events.

Before commit, compare the proposal's dependency revisions with current state. Either reject it, recompute it, or apply a specifically supported incremental update.

Do not merely blend a stale result toward the current world. That can move deleted matter back into existence or overwrite a new structure's support state.

Coalescing edits and prioritizing localized work can reduce repeated invalidation. It does not remove the need for the check.

### 13.6 Seed plus edits is no longer the entire save format

The current procedural approach remains useful. Its sparse overlay must expand to include:

- Patch descriptors and exceptional interfaces.
- Procedural provenance and ownership masks.
- Material-amount changes.
- Assembly poses and velocities.
- Persistent deformation and constitutive history.
- Broken or added mechanical connections.
- Layout replacements and required reference translations.
- Simulation time, law fingerprints, and generation versions.

Untouched regions can remain procedural. Heavily changed regions may require increasingly detailed persistent state.

There is no general guarantee that arbitrarily destructive play remains a tiny overlay forever. Periodic snapshot compaction can reorganize the data, but cannot eliminate the information content of arbitrary changes.

### 13.7 Determinism should be scoped carefully

Reproducible generation is valuable. Bitwise cross-platform nonlinear simulation is a much stronger requirement.

Even established physics libraries document conditions concerning operation order, numerical functions, and build features. Floating-point arithmetic does not become universally deterministic merely because the source language is Rust. [S26]

**Recommendation:** use server-authoritative committed state, deterministic ordering where practical, and explicit numerical/law versions. Make client prediction repairable.

Test CPU/GPU mapping agreement separately. A client renderer should not derive different boundary positions from an allegedly identical descriptor.

### 13.8 Native compiled mods belong in the world contract

Because PWC compiles mods into its binaries, a world should retain a simulation fingerprint covering the relevant material, generator, geometry, and physics rules.

The client and headless server do not need identical executable files. They do need compatible world semantics and protocol interpretation.

A purely visual mod should not unnecessarily invalidate a save. A mod changing mass, material response, or topology behavior should require an explicit compatibility decision or migration.

The package manager resolves and builds the chosen code. The runtime still owns the world's authoritative law identifiers and validates compatibility.

## 14. An unbounded procedural world needs finite rules

### 14.1 Addressability is not simulation completeness

An effectively unbounded coordinate space does not mean every region has to be expanded into voxels or simulated at full resolution.

However, a mass that exists procedurally should not disappear gravitationally because nobody is looking at it.

Separate:

- Whether matter is part of the world.
- How compactly it is represented.
- At what accuracy it is simulated.
- Whether detailed visual chunks are resident.

### 14.2 Infinite Newtonian gravity needs boundary conditions

For a finite isolated system, the potential can be referenced to zero at infinity.

For an arbitrary infinite distribution with nonzero mean density, naive summation is not a complete definition. Convergence and the treatment of the background depend on additional assumptions.

Possible product rules are:

| Universe rule | Strength | Cost |
|---|---|---|
| Finite isolated physical system inside a huge address space | Simple and well-defined | Not an indefinitely populated Newtonian universe |
| Periodic domain with a defined background convention | Established simulation formulation | Repetition and cosmological conventions enter gameplay |
| Analytically specified far field plus finite detailed regions | Efficient where the generator supports it | Far-field model must remain consistent as the world changes |
| Finite-range attraction with a smooth transition | Uniform, bounded-neighborhood game rule | Deliberately differs from Newtonian gravity at great distances |

**Recommendation:** use a finite isolated test system first. For an indefinitely populated game universe, I favor an explicit, extremely long but finite interaction range unless exact large-scale Newtonian behavior becomes a genuine gameplay requirement.

That is a proposed fictional law, not something an octree grants for free. Do not introduce it silently.

### 14.3 If a finite range is chosen, define it through a consistent force law

One possibility is a symmetric radial kernel

$$
g_{j\to i}
=
-Gm_j\,k(r)\,\widehat r,
$$

where `r̂` points from source `j` to target `i` and

$$
k(r)=\frac{r}{(r^2+\varepsilon^2)^{3/2}}\,w(r).
$$

The window `w` is one throughout the ordinary gameplay range, then decreases smoothly to zero before a fixed cutoff `R_g`.

A consistent source potential is

$$
\Phi_j(r)
=
-Gm_j\int_r^{R_g}k(s)\,ds,
\qquad r<R_g,
$$

and zero outside. Its negative gradient produces the chosen force.

Defining a tapered force and integrating it avoids accidentally adding an unintended transition-shell force by multiplying the potential and forgetting the derivative of the window.

Use the same pairwise rule in both directions. A cutoff based on each player's render distance would violate the intended shared world.

### 14.4 Loading must not act as a physical cause

Generating a detailed chunk should reveal a refined description of existing matter. It should not create a new pulse of gravity, start an overdue reaction cascade at arbitrary wall-clock time, or suddenly attach a mechanical support.

If refinement changes an approximation, bound and manage that error as a numerical update.

Similarly, unloading should not remove an orbiting body's gravity or freeze a dangerous nearby collision.

### 14.5 Sleeping and coarse simulation need physical criteria

A stable terrain region can remain asleep if its residual forces, predicted motion, and coupling to active regions remain below suitable thresholds.

A moving assembly may be represented by a coarse shape and orbit state while distant. It still advances in simulation time.

Wake regions when changing forces, predicted contact, broken supports, or other physical events invalidate their approximation. Player proximity may demand more detail, but should not determine whether the underlying event occurs at all.

A fully faithful unbounded dynamic universe is not computationally available. The intended guarantee is controlled local behavior under explicit approximation rules.

## 15. Performance and scheduling

### 15.1 A universe-wide uniform lattice is not the right storage model

Consider a planet with radius `10^7 m`. Sampling its bounding cube every 256 meters requires approximately

$$
\left(\frac{2\times10^7}{256}\right)^3
\approx 4.77\times10^{14}
$$

sample locations.

At 48 bytes per location, that is about **22.9 petabytes**, before indexing or solver overhead. This is a scale calculation, not an estimate of PWC's current memory use.

The useful version of your coarse-lattice idea is therefore a hierarchy with local lattices where needed, not one uniform lattice covering every address.

### 15.2 Make each subsystem pay for its own detail

| Subsystem | Refine when | May remain coarse when |
|---|---|---|
| Mass hierarchy | Field error is too large near a query | Distant aggregate moments are sufficient |
| Mechanical mesh | Supports, voids, material contrast, or strain need detail | The same effective response is adequately represented |
| Collision | Small surfaces or fast relative motion matter | A conservative approximation meets the contact tolerance |
| Render mesh | Surface/screen error becomes visible | Distant geometry can be simplified |
| Material storage | Edits or reactions require explicit state | Procedural or compressed state is sufficient |

Do not force all five to use the same chunk size or refinement tree.

Share spatial indexing infrastructure where helpful, but avoid assuming that one hierarchy's approximation criteria are suitable for every other task.

### 15.3 Coarse deformation is not automatically cheap

A `17 × 17 × 17` control lattice contains 4,913 nodes. Three `f64` displacement components alone consume about 115 KiB. Velocities, material state, connectivity, collision data, and solver workspaces add more.

This is reasonable for a bounded number of active regions. It is not reasonable to allocate it indiscriminately for every potential chunk.

Unchanged rigid or analytically mapped patches should use compact descriptors. Allocate deformation detail on demand and retire it only when the state can be represented faithfully in a simpler form.

### 15.4 The headless path is a requirement

The public project direction includes servers without a GPU. Therefore, authoritative correctness should not depend on a Vulkan compute device being available.

GPU acceleration can be an optional implementation of selected workloads. It should be justified by the actual workload, including transfer, synchronization, and validation costs.

Your Vulkan experience is relevant, but a coarse irregular solver may first benefit more from good data layout, bounded active regions, and a suitable algorithm than from moving everything to the GPU.

Do not use the current renderer's frame rate as evidence that planetary mechanics will be inexpensive.

### 15.5 Budget the worst cases

Track at least:

- Source-query counts and traversal depth.
- Active mechanical degrees of freedom.
- Solver iterations and failed proposals.
- Contact candidates and continuous-collision work.
- Cells invalidated by one edit.
- Fracture connectivity work.
- Remapping transfer size.
- Geometry bytes transmitted per second.
- Save growth under sustained construction and destruction.

A thin cut through a large structure or many simultaneous edits can invalidate substantial work. Rarity is not a scheduling strategy.

Background jobs should be cancellable, revision-aware, and bounded. The live server needs a defined behavior when a proposal takes longer than expected.

### 15.6 Enormous radius also changes the physical scale

For a uniform sphere,

$$
g_{\rm surface}=\frac{4\pi}{3}G\rho R.
$$

At fixed density and gravitational constant, making the radius larger also increases surface gravity.

For the illustrative `R = 10^7 m` and `ρ = 3000 kg/m³`, ordinary `G` gives about `8.39 m/s²`. Making that same-density world 100 times larger gives 100 times the surface gravity.

You cannot independently demand any radius, any density, and Earth-like surface acceleration while retaining that exact force law.

Choose a coherent fictional unit system and calibrate it. A universal game gravitational constant is consistent with your philosophy. A different hidden constant for every detected planet is not.

Likewise, an arbitrarily high material strength can delay deformation, but the stress scale grows with size. “Strong enough for every possible construction” effectively means rigidity, which is a separate limiting model.

### 15.7 Use error budgets to choose update frequency

If a neglected change in acceleration is bounded by `Δa` for time `Δt`, its positional effect is roughly bounded by

$$
\Delta x\lesssim \frac12\Delta a\,\Delta t^2
$$

under the corresponding simple assumptions.

This helps connect source-update accuracy with gameplay tolerance.

It does not justify using one universal multi-hour update interval. Nearby fast-moving mass can require frequent updates while a stable far field can remain cached.

For deformation, bound surface displacement, angular change, and strain per accepted increment. These are easier to relate to playability than a rule that recomputes all warping at a fixed wall-clock time.

## 16. Code organization and interfaces

### 16.1 Keep the first implementation in the existing project

The currently visible organization already distinguishes the game, a material crate, and a sibling renderer. Build on that rather than creating a separate repository for every proposed subsystem.

Suggested destinations:

| Location | Responsibility |
|---|---|
| Existing material crate | Configuration identity and the eventual mechanical-response bridge |
| `src/space/` | Universe positions, local frames, patch and cell addressing |
| `src/geometry/` | Embeddings, adjacency, physical bounds, geometry validation |
| `src/gravity/` | Source summaries, field queries, approximation policy |
| `src/mechanics/` | Assemblies, connections, coarse solves, constitutive history |
| `src/world/` or equivalent existing area | Authoritative matter ownership and transactions |
| Existing networking/save areas | Revisions, snapshots, events, compatibility |
| `examples/planet_lab/` or a development binary | Small geometry and physics experiments |
| Existing `voxel-engine` repository | Rendering support for mapped/tessellated patch surfaces and relative coordinates |

These are proposed modules. Reuse the real existing organization after a source audit rather than mechanically creating all these folders.

Extract a pure geometry or mechanics crate when its interfaces become stable and another consumer benefits. A separate repository is justified by genuine independent reuse, not by the ambition of the feature.

### 16.2 Give the core enforceable boundaries

Useful interfaces include:

~~~rust
trait GeometrySnapshot {
    fn position(&self, point: PatchPoint) -> UniversePosition;
    fn jacobian(&self, point: PatchPoint) -> DMat3;
    fn face_connections(&self, cell: CellHandle, face: Face)
        -> FaceConnections;
    fn physical_bounds(&self, region: PatchRegion) -> ConservativeBounds;
}

trait MassOracle {
    fn summarize(&self, region: SourceRegion, accuracy: Accuracy)
        -> MassSummaryResult;
}

trait GravityQuery {
    fn sample(&self, position: UniversePosition, accuracy: Accuracy)
        -> GravitySample;
}

trait MechanicsSolver {
    fn propose(&self, snapshot: &MechanicalSnapshot, step: StepRequest)
        -> Result<MechanicalProposal, SolveFailure>;
}
~~~

These illustrate responsibilities. A real hot path may use concrete types, batched kernels, and enums rather than a virtual call per block or source.

The commit layer validates proposals and owns live-world mutation. Solvers should not each invent their own persistence, handle translation, or cache invalidation semantics.

### 16.3 Keep policy replaceable and invariants central

Suitable replaceable policies include:

- Material-to-mechanics parameter mapping.
- Gravity kernel and approximation method.
- Geological timescale.
- Layout generation preferences.
- Fracture law.
- Optional fluid/particle backend.

The shared core should define:

- Ownership and amount accounting.
- Address and revision semantics.
- Geometry validity contracts.
- Authoritative transaction publication.
- Contact and attachment interoperability.

This is how a minimal core can remain powerful: it supplies the rules for composing mechanisms safely, while allowing the mechanisms themselves to evolve.

### 16.4 The dependency cycle should pass through snapshots

~~~mermaid
flowchart TD
    W["Committed world state"] --> M["Physical mass summaries"]
    M --> G["Gravity queries"]
    W --> S["Mechanical snapshot"]
    G --> S
    S --> P["Motion and fracture proposal"]
    P --> V["Geometry and conservation validation"]
    V --> C["Atomic commit"]
    C --> W
    W --> R["Rendering and player queries"]
~~~

The cycle is real because moved matter changes gravity. Snapshots and commits make it tractable without pretending that gravity and geometry are independent.

Layout generation is another proposal producer. It must pass the same ownership and publication rules, with additional material-transfer validation when addresses change.

## 17. A staged implementation plan with exit criteria

### Stage 0 — Establish the geometric limit

Create a small standalone planet laboratory before rewriting the world.

Implement a few known mappings, place the same simple building at the face center, edge, corner, and several depths, and inspect both the building and per-cell metrics.

Include a central-region experiment. Teleporting around the prototype is essential; waiting to encounter rare locations would hide the important failures.

**Exit criterion:** choose and document the actual tolerated distortion and exceptional connectivity. If no tested layout makes ordinary building pleasant, revise the representation before building the rest.

### Stage 1 — Define identity and conservation

Introduce the distinction between cell address, material configuration, amount, reference geometry, and physical position.

Keep the existing world Cartesian while these contracts are exercised. Confirm how the reaction candidate interacts with mass.

**Exit criterion:** edits, loading, reactions under the chosen model, and rigid rebasing preserve the declared conserved quantities. No mass rule remains implicit.

### Stage 2 — Implement matter-derived gravity independently

Start with a finite set of extended or sampled sources and a direct reference evaluator. Add the tree approximation, near-field treatment, and procedural summaries.

Use uniform spheres, cavities, two masses, and deliberately irregular sources.

**Exit criterion:** field error is measured against a reference over surface, interior, near-zero, and distant points. Refinement and loading produce controlled results without double counting.

### Stage 3 — Make one curved world properly playable

Integrate the selected stable atlas, arbitrary-direction player control, physical raycasting, collision, geometry epochs, and camera-relative rendering.

The world's mass comes from matter. Its initial geometry is generated rather than dynamically re-optimized.

**Exit criterion:** two clients can build and move across supported interfaces, save, reload, and reproduce the same physical world. Declared interior limits are visible and documented.

This stage is already a substantial feature. It should be useful independently of planetary relaxation.

### Stage 4 — Add rigid assemblies and ordinary splitting

Allow bounded material regions to move while retaining their local grids. Support detachment, rotation, docking/contact, and explicit bonds.

**Exit criterion:** physical position, mass, and momentum survive reparenting and splitting. Source contributions follow the moving material.

### Stage 5 — Add bounded elastic deformation

Implement the coarse mechanical prototype with one material model and validated geometry increments.

Test free fall, supported beams, inclusions, compression, rotation, and cavity boundaries.

**Exit criterion:** results converge appropriately under selected timestep/mesh refinement, and failure handling never commits invalid geometry.

### Stage 6 — Add material-dependent yielding and fracture

Choose a specific plasticity or creep model. Add irreversible state and a carefully scoped fracture law.

Use small synthetic systems first. Only then test large self-gravitating aggregates.

**Exit criterion:** weak and strong constructions differ for the intended mechanical reasons, not because of chunking or arbitrary body classification. Conservation and dissipation remain explainable.

### Stage 7 — Investigate retiling and severe flow

Implement one conservative transfer operation before attempting general automatic gravity-aligned remeshing.

For example, retile a simple known deforming region while preserving material composition and an attached entity.

**Exit criterion:** quantitative errors in occupancy, mass, momentum, state, and attachments stay within agreed limits. Thin structures and repeated operations do not disappear or accumulate unacceptable drift.

Failure here should narrow the advertised flow capability. It need not invalidate stable curved terrain or ordinary assemblies.

### Stage 8 — Generalize formation and long-lived worlds

Only after the previous gates should arbitrary large clusters form new buildable curved layouts automatically.

Integrate background scheduling, far-field policy, long saves, and sustained multiplayer modification.

**Exit criterion:** demonstrate the entire lifecycle—formation, construction, deformation, fracture, separation, and continued building—under a realistic server budget.

No benchmark in this document establishes that final gate.

## 18. Adversarial tests that expose architectural mistakes

| Test | Expected result | Mistake it exposes |
|---|---|---|
| One ordinary isolated block | Negligible self-deformation; finite local field model | Tiny objects forced into spherical layouts |
| Small ship in nearly uniform gravity | Bulk free fall with little internal gravitational strain | Gravity magnitude used as warp strength |
| Same ship on supports | Support-dependent stress | Ignoring contact and reaction forces |
| Uniform sphere, sampled through center | Correct interior profile for the chosen kernel; zero at the symmetric center | Treating extended mass as one point everywhere |
| Hollow spherical shell | Zero cavity field for exact inverse-square gravity; quantified kernel deviations otherwise | Confusing force-law error with numerical error |
| Equal masses with a midpoint query | Cancellation without undefined orientation math | Selecting one attractor or normalizing zero |
| Huge weak cube | Model-dependent relaxation, with visible representation limits | A classifier replacing mechanics |
| Huge strong cube | Persists within its strength range, with stresses accounted for | Unconditional “large means spherical” rule |
| Long weak beam | May bend or fail despite low compactness | Compactness exemption mistaken for physics |
| Strong inclusion in weak matrix | Coupled stress, possible interface failure | Independent per-material vertex warps |
| Deep cavity in compressed matter | New boundary conditions and local load redistribution | Hydrostatic pressure treated as irrelevant |
| Mining a one-cell cut through a support | Load path disappears when physically severed | Invisible coarse-mesh bridges |
| Two separated pieces | Can move independently | A continuous global field still binding them |
| Two bodies touching | Contact without automatic welding | Gravity or overlap used as ownership |
| A toroidal structure | Preserves its hole until actual material closes it | Forced sphere topology |
| Cube-sphere face corner | Measured skew and correct adjacency | Radius assumed to remove map distortion |
| Path into the central region | No radial collapse; clear grid transition | Pressure policy mistaken for a coordinate fix |
| Very tall construction | Defined cell-size/patch policy | Infinite widening of angular columns |
| Fast passage near a moving mass | Timely gravitational update | Only block edits invalidating gravity |
| Rigid 180° rotation | No intermediate collapse | Linear blending of positions |
| Self-contacting deformable body | No unintended crossing | Positive local determinant treated as global safety |
| Thin wall during retiling | Conserved occupancy and composition within tolerance | Sampling erasing small features |
| Repeated split and merge | No accumulated amount creation | Rounding and ownership double counting |
| Loading in a different order | Same declared world state | Generation acting as a physical event |
| Two clients straddling a seam | Shared selection, collision, and placement | Independent chart ownership |
| Save during a background solve | Last committed state restores coherently | Partially committed deformation |
| Different render LOD settings | Identical authoritative physics | Graphics resolution leaking into simulation |
| Modified material-law mod | Explicit compatibility or migration | World semantics changing unnoticed |

For approximate quantities, record absolute error as well as relative error. Relative gravity error alone is misleading near a true zero.

A test suite should include deliberately invalid geometry and stale transactions. Only testing pleasant spherical examples would leave the hardest promises unexamined.

## 19. Decisions to settle before production integration

| Decision | My recommended starting choice | What could change it |
|---|---|---|
| What is fundamental: a block or its matter? | Matter amounts are authoritative; cells are persistent but replaceable representation | A deliberate priority for everlasting individually shaped blocks |
| How does material identity affect mass? | Conserved amounts independent of mutable resource coordinates | A separately designed reaction mass/energy law |
| Must every location feel exactly like flat Minecraft? | Nearly cubic ordinary regions, with explicit exceptional topology | A decision to sacrifice global curvature or unrestricted interiors |
| Can a construction remain permanently rigid? | Rigidity is a supported mechanical limit, not a planet exception | Desire for every object to have finite strength |
| What is the initial interior layout? | Prototype a finite core plus mapped transition blocks | Measured quality favoring adaptive shells or another atlas |
| What makes material detach? | Connectivity and mechanical failure | A deliberately different fictional rule |
| How much severe flow is required initially? | Defer unrestricted retiling; preserve useful solids first | A game loop centered on flowing planetary matter |
| What happens at immense distances? | Finite isolated prototype; explicit finite-range law for an unbounded populated universe | Genuine need for a cosmological Newtonian model |
| Must clients reproduce every solve? | Server-authoritative results with repairable prediction | A demonstrated need for deterministic lockstep |
| Where do algorithms live? | Modules in the existing project until interfaces mature | Real reuse requiring independent crates or repositories |

The essential architectural commitment is the separation of **material state, coordinate layout, physical motion, and approximation**. It gives PWC room for strong artificial structures and emergent worlds without promising that one warp formula can satisfy mutually conflicting requirements.

## Appendix A. Independent mathematical checks

These derivations support the critique. They are not measurements of an implemented PWC system.

### A.1 Curvature and map distortion are different derivatives

For the normalized face map

$$
p=R\frac{(u,v,1)}{\sqrt{s}},
\qquad s=1+u^2+v^2,
$$

differentiation gives

$$
p_u=R\frac{(1+v^2,-uv,-u)}{s^{3/2}},
$$

$$
p_v=R\frac{(-uv,1+u^2,-v)}{s^{3/2}}.
$$

Consequently,

$$
\|p_u\|=\frac{R\sqrt{1+v^2}}{s},
\qquad
\|p_v\|=\frac{R\sqrt{1+u^2}}{s},
$$

and

$$
p_u\cdot p_v=-\frac{R^2uv}{s^2}.
$$

The angle formula in Section 3 follows directly.

For an equal parameter step, the face-corner tangent length relative to the face-center tangent length is

$$
\frac{\sqrt2}{3}\approx0.4714.
$$

The local area factor is

$$
\|p_u\times p_v\|=\frac{R^2}{s^{3/2}},
$$

so the corner-to-center area ratio is

$$
\frac{1}{3\sqrt3}\approx0.19245.
$$

These particular numbers apply to this projection and equal parameter spacing. They do not say that every cube-sphere mapping has these distortions.

They do prove that increasing `R` alone does not fix this one.

### A.2 Why spherical quad layouts contain exceptional vertices

For a closed quadrilateral mesh, each face has four edges and every edge belongs to two faces:

$$
4F=2E.
$$

For a sphere,

$$
V-E+F=2.
$$

Substituting `F=E/2` gives `4V-2E=8`.

Since the sum of vertex degrees is `2E`,

$$
\sum_v(4-d_v)=8.
$$

An everywhere-valence-four mesh would give zero, which contradicts the sphere's topology.

This argument concerns the connectivity of a closed quadrilateral surface. It is not an argument against a three-dimensional Cartesian coordinate system existing around a spherical object.

### A.3 Why columns become thin near the center

At radius `r`, an angular interval `Δθ` has arc length

$$
\ell=r\Delta\theta.
$$

Keeping the same angular cells while reducing `r` therefore reduces their widths in direct proportion.

Making the surface radius enormous only postpones the problem if the player remains in a shallow shell. It does not remove the origin.

A finite core or changing resolution is a representation change, independent of the value of pressure at that location.

### A.4 Gravity and pressure inside a uniform sphere

Let density be constant, and ignore rotation.

The enclosed mass at radius `r` is

$$
M(r)=\frac{4\pi}{3}\rho r^3.
$$

Spherical symmetry then gives the inward acceleration magnitude

$$
g(r)=\frac{GM(r)}{r^2}
=\frac{4\pi}{3}G\rho r.
$$

Hydrostatic balance requires

$$
\frac{dp}{dr}=-\rho g(r).
$$

With zero external surface pressure, `p(R)=0`, integration gives

$$
p(r)=\frac{2\pi}{3}G\rho^2(R^2-r^2).
$$

Thus `g(0)=0` while pressure is largest at the center.

For `R=10^7 m` and `ρ=3000 kg/m³`:

| Quantity | Calculated value |
|---|---:|
| Surface acceleration | 8.387 m/s² |
| Center pressure | 1.258 × 10¹¹ Pa |
| Center pressure in GPa | 125.8 |

This is a constant-density Newtonian model. It is not a realistic detailed equation of state for an actual planet.

### A.5 Tiny angle does not imply negligible displacement over a large face

At `R=10^7 m`, the normal rotates across a one-meter block by approximately

$$
10^{-7}\ \text{radians}
\approx 0.00000573^\circ.
$$

Across a 1000-meter span it rotates by approximately `0.00573°`.

Yet the midpoint deviation between a 1000-meter chord and the spherical surface is about 1.25 centimeters:

$$
R-\sqrt{R^2-(L/2)^2}
\approx\frac{L^2}{8R}.
$$

A normal-angle test and a surface-position test measure different aspects of approximation quality.

### A.6 Why arbitrarily chosen local rotations do not define a warp

If a smooth map `χ(X)` exists and

$$
F_{ij}=\frac{\partial\chi_i}{\partial X_j},
$$

then mixed partial derivatives must agree:

$$
\frac{\partial F_{ij}}{\partial X_k}
=
\frac{\partial F_{ik}}{\partial X_j}.
$$

Choosing a rotation independently at every point generally does not satisfy this compatibility condition.

One intuitive test is to walk around a small closed loop using the proposed local coordinate steps. If the steps do not close, they do not describe one ordinary consistent coordinate chart.

Gravity can define a preferred vertical direction without resolving this problem for a full three-axis grid.

### A.7 Why mass changes with the determinant only through density

For reference volume `dV₀` and current volume `dV`,

$$
dV=J\,dV_0.
$$

Conservation of material mass gives

$$
\rho\,dV=\rho_0\,dV_0,
$$

hence

$$
\rho=\frac{\rho_0}{J}.
$$

This is why deformation should not recompute a material parcel's mass from a fixed density and its changed volume.

Reference density, amount, deformation, and current density must remain distinct.

### A.8 Large radius improves one problem while worsening another

The local curvature scale decreases as `1/R`.

At fixed density and ordinary Newtonian gravity:

- Surface gravity increases as `R`.
- Characteristic gravitational pressure increases as `R²`.
- The number of equal-sized surface cells increases as `R²`.
- A fully explicit volume increases as `R³`.

Thus “make the world enormous” helps local flatness while increasing physical and representational demands. A good architecture takes advantage of the first fact without ignoring the others.

## Appendix B. Sources and how they were used

Public sources were consulted on 3 October 2026. Links point to primary papers, author or research-group pages, official documentation, and the project's public repository. Publisher abstracts were used only for claims supported by those abstracts.

The architecture, priorities, interfaces, prototype thresholds, and acceptance gates in this document are my synthesis. The cited authors did not propose or validate this exact PWC design.

### Project grounding

| ID | Source | Use and limit |
|---|---|---|
| S1 | [Project Watt Cubed public repository and README][S1] | Procedural multiplayer, authoritative headless server, and renderer separation. Not a complete code audit. |
| S2 | [Project Watt Cubed Cargo manifest][S2] | Visible material workspace crate and sibling renderer dependency. Proposed directories were not inferred to exist. |
| P1 | `watt_cubed_reaction_specification.md`, 11 September 2026 | Read its state/scope definition and later limitations. Establishes that the candidate preserves constituent counts while leaving mechanics and physical energy undefined. It is a proposal, not proof of current implementation. |
| P2 | `Emergent Planets, Curved Voxels, and Matter-Derived Gravity in PWC`, 3 October 2026 | Read the opening proposal and its cube-sphere argument. Treated as earlier design work to critique, not as external validation. |
| P3 | Your supplied conversation and recovered prior PWC statements | Grounds the preference for generic rules, deep building, multiplayer, and a minimal capable core. Prior assistant suggestions were not treated as your confirmed decisions. |

### Spherical grids and volume layouts

| ID | Source | Use and limit |
|---|---|---|
| S3 | Ronchi, Iacono, Paolucci, [The “Cubed Sphere”: A New Method for the Solution of Partial Differential Equations in Spherical Geometry][S3], 1996 | Foundational six-region construction. Does not imply distortion-free voxel building. |
| S4 | [Finite-volume transport on various cubed-sphere grids][S4], 2007 | Orthogonality, uniformity, and corner tradeoffs. Publisher abstract/preview consulted. |
| S5 | Pietroni et al., [Hex-Mesh Generation and Processing: a Survey][S5], 2022 preprint | Distinguishes topology, frame fields, parameterization, mesh extraction, and validity. Used as a map of the problem, not a blanket claim that later work made no progress. |
| S6 | Liu and Bommes, [Locally Meshable Frame Fields][S6], 2023 | Local meshability is a separate condition; the authors explicitly distinguish it from full meshability. |
| S15 | Johnen, Weill, Remacle, [Robust and efficient validation of the linear hexahedral element][S15], 2017 | Why sparse Jacobian sampling can miss invalid hexahedra; methods based on polynomial bounds. |
| S22 | Brückler, Bommes, Campen, [Integer-Sheet-Pump Quantization for Hexahedral Meshing][S22], 2024 | Integer layout decisions are an explicit stage, not a side effect of a smooth field. |
| S23 | Brückler and Campen, [Volume Quantization with Flexible Singularities for Hexahedral Meshing][S23], 2026 | Recent progress allowing singularity changes during quantization. Does not supply PWC's persistent state-transfer semantics. |
| S24 | Yu et al., [NeurFrame: Learning Continuous Frame Fields for Structured Mesh Generation][S24], March 2026 preprint | Recent continuous frame-field approach. Treated as research, not a required or validated game-engine component. |

### Gravity and mechanics

| ID | Source | Use and limit |
|---|---|---|
| S7 | Piet Hut, [Algorithms][S7], with Barnes and Hut's [1986 Nature paper](https://www.nature.com/articles/324446a0) | Author account of hierarchical force calculation. Complexity statements are not universal worst-case guarantees. |
| S8 | Richard Fitzpatrick, [Potential due to uniform sphere][S8], University of Texas | Extended spherical source behavior; Appendix A independently derives acceleration and pressure for the chosen idealization. |
| S9 | Allan F. Bower, [Applied Mechanics of Solids, §2.2: Mathematical Description of Shape Changes][S9] | Reference/current geometry and deformation measures. |
| S10 | Bower, [§2.4: Equations of Motion][S10] | Conservation and the balance law underlying stress, supports, and acceleration. |
| S11 | Bower, [§3.5: Hyperelastic Materials][S11] | Energy-based constitutive models and frame indifference. Not evidence that the chosen elastic law models all PWC materials. |
| S12 | Bower, [§3.7: Small Strain Plasticity][S12] | Yielding and the distinction between reversible and irreversible response. |
| S13 | Bower, [§3.11: Soils][S13] | Pressure-sensitive strength and compaction; cautions against treating every material like a metal. |
| S14 | Macklin, Müller, Chentanez, [XPBD: Position-Based Simulation of Compliant Constrained Dynamics][S14], 2016 | Candidate constraint method and its convergence limitations. |
| S16 | Li et al., [Incremental Potential Contact][S16], 2020 | Feasibility-preserving contact/inversion research under a specific numerical framework. No claim that its guarantees transfer automatically to arbitrary patches. |
| S18 | Stomakhin et al., [A Material Point Method for Snow Simulation][S18], 2013 | Alternative representation for large material deformation. Does not establish persistent voxel construction semantics. |
| S19 | Taubin, [A Signal Processing Approach to Fair Surface Design][S19], 1995 | Surface fairing as geometry processing, distinct from the proposed material mechanics. |
| S20 | Dehnen, [A Very Fast and Momentum-Conserving Tree Code][S20], 2000 | Symmetric gravitational interactions and linear momentum accounting. |
| S21 | Springel et al., [GADGET-4 code paper][S21], 2021 | Hierarchical forces, timesteps, and large dynamic ranges. Its simulations are not a PWC performance benchmark. |

### Engine and game precedents

| ID | Source | Use and limit |
|---|---|---|
| S17 | Bowerbyte, [Blocky Planet — Making Minecraft Spherical][S17] | The closest directly relevant development account: projection distortion, shells, transitions, and stated core limitations. |
| S25 | Godot, [Large world coordinates][S25] | Official explanation of precision problems and rendering considerations. PWC need not adopt Godot. |
| S26 | Rapier, [Determinism][S26] | Official example of the conditions behind cross-platform determinism. PWC need not adopt Rapier. |

[S1]: https://github.com/gusahlg/project_watt_cubed
[S2]: https://github.com/gusahlg/project_watt_cubed/blob/main/Cargo.toml
[S3]: https://www.sciencedirect.com/science/article/pii/S0021999196900479
[S4]: https://www.sciencedirect.com/science/article/abs/pii/S0021999107003105
[S5]: https://arxiv.org/html/2202.12670v1
[S6]: https://www.algohex.eu/publications/locally-meshable-frame-fields/
[S7]: https://www.ias.edu/piet/act/comp/algorithms
[S8]: https://farside.ph.utexas.edu/teaching/celestial/Celestial/node18.html
[S9]: https://solidmechanics.org/Text/Chapter2_2/Chapter2_2.php
[S10]: https://solidmechanics.org/Text/Chapter2_4/Chapter2_4.php
[S11]: https://solidmechanics.org/Text/Chapter3_5/Chapter3_5.php
[S12]: https://solidmechanics.org/Text/Chapter3_7/Chapter3_7.php
[S13]: https://solidmechanics.org/Text/Chapter3_11/Chapter3_11.php
[S14]: https://matthias-research.github.io/pages/publications/XPBD.pdf
[S15]: https://arxiv.org/abs/1706.01613
[S16]: https://ipc-sim.github.io/
[S17]: https://bowerbyte.com/posts/blocky-planet/
[S18]: https://www.disneyanimation.com/publications/a-material-point-method-for-snow-simulation/
[S19]: https://research.ibm.com/publications/signal-processing-approach-to-fair-surface-design
[S20]: https://arxiv.org/abs/astro-ph/0003209
[S21]: https://wwwmpa.mpa-garching.mpg.de/gadget4/gadget4-code-paper.pdf
[S22]: https://www.algohex.eu/publications/integer-sheet-pump-quantization-for-hexahedral-meshing/
[S23]: https://onlinelibrary.wiley.com/doi/10.1111/cgf.70349
[S24]: https://arxiv.org/abs/2603.12820
[S25]: https://docs.godotengine.org/en/stable/tutorials/physics/large_world_coordinates.html
[S26]: https://rapier.rs/docs/user_guides/rust/determinism/

## Appendix C. Verification, confidence, and limits

### What was checked for this report

- Read the supplied design conversation and recovered PWC philosophy statements.
- Read the relevant scope and limitation sections of the saved reaction candidate.
- Read the opening of the earlier planet research report.
- Inspected the available public README and Cargo manifest.
- Consulted the primary sources listed above, including recent geometry research.
- Derived the projection-angle, topology, pressure, density, and scale relationships.
- Ran numerical arithmetic checks for the projection corner, gravity/pressure example, rough stress scale, uniform-grid memory estimate, control-lattice storage, and chord error.
- Reviewed the proposed separation of source ownership, physical motion, layout changes, and committed multiplayer state.

### What was not established

- No PWC game code was changed.
- No planet prototype, FEM solver, gravity tree, or networking extension was implemented.
- No claimed FPS, server capacity, or convergence rate was benchmarked.
- No automatic mesher was demonstrated on PWC constructions.
- No exact material-property bridge was derived from the current reaction score.
- No proof establishes that every desired player construction fits the recommended representation.
- The complete public source tree was not audited; repository integration needs that next.

### Confidence by claim

| Claim | Assessment |
|---|---|
| Large radius does not remove projection skew | Established; explicit derivation |
| A regular spherical quad layout needs exceptional topology | Established under the stated mesh assumptions |
| Gravity magnitude alone does not determine deformation | Established mechanics |
| A continuous one-to-one deformation cannot tear connected material | Established topology |
| Positive local determinants alone do not rule out global overlap | Established geometric distinction |
| Matter, geometry, and mechanics should have separate interfaces | Strong engineering recommendation |
| A stable curved-world prototype is worth building | Strong recommendation, subject to gameplay evaluation |
| Coarse deformation can preserve acceptable building at useful cost | Plausible, requires measurement |
| Arbitrary emergent worlds can be automatically retiled without harming constructions | Major research risk |
| This exact architecture will scale to unrestricted multiplayer planetary destruction | Unproven |

The best next deliverable is a small, instrumented geometry laboratory with ugly cases deliberately included. Its job is to determine which compromises preserve the PWC experience before the larger architecture becomes expensive to change.
