# AI Image-Generation Prompts for CableTract Manuscript

These prompts produce visuals that complement the TikZ schematics in the
manuscript. Use whichever image model you prefer (Midjourney, DALL-E,
Imagen, FLUX, etc.) — the prompts are written to be model-agnostic.

When the rendered images are ready, save them into the `figures/` folder
with these exact filenames so the manuscript picks them up automatically:

| File | Required filename | Used by |
|---|---|---|
| Hero scene | `figures/F0b_hero_scene.png` | §3 opening figure (currently has a placeholder box) |
| Main Unit hardware close-up | `figures/F0a1_mu_render.png` | Optional companion to the F0a TikZ schematic |
| Anchor hardware close-up | `figures/F0a2_anchor_render.png` | Optional companion to the F0a TikZ schematic |

If you want the MU/Anchor renders embedded too, tell me the filenames you
end up with and I will wire them into the manuscript.

Aspect ratios: hero image **16:9** (panoramic field shot), the two hardware
close-ups **4:3 or 1:1**.

---

## 1. F0b — Hero scene: CableTract operating in a field

**Intent.** A photo-realistic establishing shot that lets a sceptical
reviewer see what the system actually looks like in operation. No labels,
no annotations, no logos. The TikZ schematics carry all the labelling.

### Prompt

> Wide cinematic photograph, golden-hour lighting, freshly tilled brown
> agricultural field stretching to a low horizon. In the left foreground,
> a compact electric agricultural robot the size of a small utility
> vehicle: rectangular blue-grey aluminium body about 1.5 m tall, a single
> mono-crystalline silicon photovoltaic panel tilted on top, a small
> three-bladed vertical-axis wind turbine on a short mast at one corner,
> a heavy-duty winch with a black drum on the front face, two large
> heavy-duty rubber wheels on either side for transport between fields,
> AND four helical auger pile drives mounted under the chassis between
> the wheels currently planted into the soil to anchor the body during
> operation. In the right background, about 50 metres away across the
> field, a smaller blue-grey rectangular module of similar industrial
> design, no PV panel, no wind turbine, no winch, with two smaller
> wheels and nine helical auger pile drives planted into the soil and a
> redirect sheave block on top. A single taut steel cable runs through
> the air across the 50-metre span between the two modules at roughly
> 0.8 metre above the ground. Halfway along the cable, a lightweight
> orange-and-grey implement carriage with a 1.5-metre wide chisel-tine
> ground tool engaged in the soil, ploughing a clean strip behind it.
> Soft warm sunlight, shallow depth of field, no people, no other
> machinery, no buildings, clear sky with a few high cirrus clouds.
> Engineering documentary photography style, ultra-detailed, 16:9.

### Negative prompt (if your model accepts one)

> No text, no logos, no watermarks, no labels, no humans, no farm animals,
> no other tractors, no buildings, no roads, no fences, no power lines,
> no captions, no UI overlays, no driver, no cabin, no steering wheel.

### Notes

- "Cable" is the hardest element for image models — they tend to draw a
  rope on the ground or tangle multiple cables. If your first generation
  shows the cable wrong, regenerate with the words *"single straight
  taut steel aerial cable, 0.8 metres above the ground, perfectly
  straight line from the winch on the left module to the sheave on the
  right module"* explicit.
- The modules have **both** wheels (for transport between fields and for
  stepping along the headland between strips) **and** helical screw
  augers (for resisting the cable draft during operation). If the model
  shows only one or the other, add *"both rubber wheels for mobility AND
  helical screw augers planted in the soil for anchoring, side by side
  under the chassis"*.

---

## 2. F0a1 — Main Unit hardware close-up

**Intent.** A photo-realistic isometric or three-quarter view of just
the Main Unit, used as a companion to the labeled TikZ schematic in
F0a. Goal is to give the reader a tactile sense of the hardware while
the schematic provides the label inventory.

### Prompt

> Photorealistic three-quarter isometric product render of a compact
> electric agricultural robot module on bare brown soil, neutral grey
> studio backdrop fading to white. Rectangular aluminium body about
> 1.5 metres long, 1 metre wide, 1 metre tall, painted matte
> blue-grey, with rounded vertical edges and visible structural ribs.
> On top: a single rectangular mono-crystalline silicon photovoltaic
> panel tilted at about 25 degrees facing the camera, dark blue cells
> with thin silver busbars. At one rear corner of the body: a small
> three-bladed vertical-axis wind turbine on a short black aluminium
> mast about 1 metre above the body. On the front face of the body: a
> heavy-duty industrial winch with a matte-black steel drum, visible
> wound steel cable, and a chunky cylindrical brushless servo motor
> bolted to the side. Mounted under the chassis: two large heavy-duty
> rubber wheels on either side for transport mode AND four helical
> screw pile augers about 30 cm long currently deployed downward into
> the soil between the wheels for self-anchoring during operation.
> Side panels visible, suggesting access doors to the battery and
> controller compartments. Realistic materials: brushed aluminium,
> anodised blue-grey, matte black steel, black rubber tyres. Unmanned
> autonomous module — no driver cabin, no driver seat, no steering
> wheel. Ultra-detailed, soft directional studio lighting from upper
> left, shallow depth of field, 4:3 aspect.

### Negative prompt

> No text, no logos, no labels, no people, no other machinery, no
> tractor, no horse, no track-laying treads, no buildings, no farms in
> the background, no captions, no driver, no cabin, no steering wheel,
> no roof.

### Notes

- The MU has **both** wheels (transport between fields, stepping along
  the headland) AND helical augers (anchoring during a pull). Most
  image models will show one or the other; force both with the explicit
  phrasing in the prompt.
- If the model attaches a steering column or driver's seat, add
  *"unmanned autonomous module, no cabin, no driver seat, no steering
  wheel"* even more emphatically.

---

## 3. F0a2 — Anchor hardware close-up

**Intent.** Same treatment for the Anchor module. The Anchor is
visually much sparser than the MU — no PV, no wind turbine, no winch,
just a redirect sheave on top of a small body sitting on a denser
auger array.

### Prompt

> Photorealistic three-quarter isometric product render of a small
> passive agricultural anchor module on bare brown soil, neutral grey
> studio backdrop fading to white. Compact rectangular aluminium body
> about 0.9 metres long, 0.6 metres wide, 0.7 metres tall, painted
> matte blue-grey to match the main unit, with rounded vertical edges.
> On top of the body: a heavy-duty cable redirect sheave block — a
> single grooved steel pulley about 15 cm in diameter mounted in a
> bracket bolted to the body, with a single taut steel cable
> approaching from the upper-left and exiting downward into the
> sheave. No photovoltaic panel, no wind turbine, no winch on this
> module. Mounted under the chassis: two compact heavy-duty rubber
> wheels on either side for headland repositioning AND nine helical
> screw pile augers arranged in a 3-by-3 cluster between the wheels,
> each about 30 cm long, planted into the soil for anchoring.
> Realistic materials: brushed aluminium, anodised blue-grey, matte
> black steel, black rubber tyres. Unmanned passive module — no
> driver cabin, no seat. Ultra-detailed, soft directional studio
> lighting from upper left, shallow depth of field, 4:3 aspect.

### Negative prompt

> No text, no logos, no labels, no people, no other machinery, no
> tractor, no buildings, no PV panels, no wind turbines, no winch, no
> captions, no driver, no cabin.

### Notes

- The Anchor has explicitly *no* PV / wind / winch — these are the
  three things image models will reflexively add. Keep them in the
  negative prompt.
- The Anchor *does* have wheels (for headland repositioning between
  strips) AND the 9 augers (for resisting cable draft) — both must be
  visible. Force both with the explicit phrasing.
- The 9-auger cluster is visually distinctive and is the headline
  feature; if the render shows fewer than 9, regenerate with
  *"exactly nine helical screw augers in a 3 by 3 grid pattern under
  the body between the two wheels, all visibly planted in the soil"*.

---

## After generation

1. Save each image with the exact filename above into the `figures/`
   folder.
2. For F0b, the manuscript will pick it up automatically on next
   compile (the placeholder in §3 disappears).
3. For F0a1 and F0a2, ping me with the filenames so I can wire them
   alongside the F0a schematic — possible layouts: a 2-row figure
   (schematic above, photo-render below), or a 4-panel figure (MU
   schematic | MU render | Anchor schematic | Anchor render).
