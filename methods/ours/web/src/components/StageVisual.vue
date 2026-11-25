<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{ stage: string; compact?: boolean }>()

const kind = computed(() => {
  const value = (props.stage || '').toLowerCase()
  if (value.includes('complete')) return 'complete'
  if (value.includes('finaliz') || value.includes('saving')) return 'finalizing'
  if (value.includes('document')) return 'documentation'
  if (value.includes('clust')) return 'clustering'
  if (value.includes('plan')) return 'planning'
  if (value.includes('component')) return 'components'
  if (value.includes('parse')) return 'parse'
  return 'build'
})
</script>

<template>
  <div class="stage-visual" :class="[kind, { compact: props.compact }]" aria-hidden="true">
    <div class="layer nodes">
      <span class="edge e1"></span>
      <span class="edge e2"></span>
      <span class="edge e3"></span>
      <span class="node n1"></span>
      <span class="node n2"></span>
      <span class="node n3"></span>
      <span class="node n4"></span>
      <span class="node n5"></span>
    </div>

    <div class="layer parse">
      <span class="code-line c1"></span>
      <span class="code-line c2"></span>
      <span class="code-line c3"></span>
      <span class="code-line c4"></span>
      <span class="scan"></span>
    </div>

    <div class="layer components">
      <span class="block b1"></span>
      <span class="block b2"></span>
      <span class="block b3"></span>
      <span class="block b4"></span>
      <span class="block-core"></span>
    </div>

    <div class="layer clustering">
      <span class="orbit o1"><i></i></span>
      <span class="orbit o2"><i></i></span>
      <span class="orbit o3"></span>
      <span class="cluster-core">◇</span>
    </div>

    <div class="layer planning">
      <span class="plan-stem"></span>
      <span class="plan-branch p1"></span>
      <span class="plan-branch p2"></span>
      <span class="plan-branch p3"></span>
      <span class="plan-leaf l1"></span>
      <span class="plan-leaf l2"></span>
      <span class="plan-root"></span>
    </div>

    <div class="layer sheets">
      <span class="sheet s1"></span>
      <span class="sheet s2"></span>
      <span class="sheet s3"></span>
      <span class="sheet s4"></span>
    </div>

    <div class="layer finalizing">
      <span class="final-ring"></span>
      <span class="final-ring inner"></span>
      <span class="final-core">◇</span>
    </div>

    <span v-if="kind === 'complete'" class="complete-check">✓</span>
  </div>
</template>

<style scoped>
.stage-visual {
  --size: 220px;
  --stage-color: color-mix(in srgb, var(--text-color) 62%, transparent);
  --stage-strong: var(--text-color);
  --stage-soft: color-mix(in srgb, var(--text-color) 9%, transparent);
  width: var(--size);
  height: var(--size);
  position: relative;
  display: grid;
  place-items: center;
  margin: 0 auto;
  flex: 0 0 auto;
}

.stage-visual.compact {
  --size: 64px;
}

.layer {
  position: absolute;
  inset: 0;
  display: none;
}

.build .nodes,
.parse .parse,
.components .components,
.clustering .clustering,
.planning .planning,
.documentation .sheets,
.finalizing .finalizing {
  display: block;
}

.node {
  position: absolute;
  width: 9%;
  height: 9%;
  min-width: 6px;
  min-height: 6px;
  border-radius: 50%;
  background: var(--stage-color);
  animation: nodePulse 1.8s ease-in-out infinite;
}
.n1 { left: 19%; top: 37%; animation-delay: 0s; }
.n2 { left: 44%; top: 16%; animation-delay: 0.15s; }
.n3 { left: 72%; top: 31%; animation-delay: 0.3s; }
.n4 { left: 37%; top: 68%; animation-delay: 0.45s; }
.n5 { left: 70%; top: 68%; animation-delay: 0.6s; }

.edge {
  position: absolute;
  height: 1px;
  background: color-mix(in srgb, var(--stage-color) 55%, transparent);
  transform-origin: left center;
  animation: edgePulse 1.8s ease-in-out infinite;
}
.e1 { left: 24%; top: 41%; width: 26%; transform: rotate(-38deg); }
.e2 { left: 47%; top: 21%; width: 25%; transform: rotate(30deg); animation-delay: 0.2s; }
.e3 { left: 42%; top: 69%; width: 27%; transform: rotate(-4deg); animation-delay: 0.35s; }

.parse {
  inset: 20% 18%;
}

.code-line {
  position: absolute;
  left: 0;
  height: 5%;
  border-radius: 99px;
  background: color-mix(in srgb, var(--secondary-text) 22%, transparent);
}
.c1 { top: 8%; width: 62%; }
.c2 { top: 28%; width: 84%; }
.c3 { top: 48%; width: 48%; }
.c4 { top: 68%; width: 72%; }

.scan {
  position: absolute;
  left: 4%;
  right: 4%;
  top: 0;
  height: 2px;
  border-radius: 2px;
  background: var(--stage-color);
  animation: scan 2s ease-in-out infinite;
}

.block {
  position: absolute;
  width: 24%;
  height: 20%;
  border: 1px solid color-mix(in srgb, var(--stage-color) 55%, var(--border-color));
  border-radius: 7px;
  background: transparent;
  animation: blockIn 2s ease-in-out infinite;
}
.b1 { left: 13%; top: 18%; animation-delay: 0s; }
.b2 { right: 13%; top: 18%; animation-delay: 0.12s; }
.b3 { left: 13%; bottom: 18%; animation-delay: 0.24s; }
.b4 { right: 13%; bottom: 18%; animation-delay: 0.36s; }

.block-core {
  position: absolute;
  width: 12%;
  height: 12%;
  border-radius: 50%;
  background: var(--stage-strong);
  animation: corePulse 1.2s ease-in-out infinite;
}

.orbit {
  position: absolute;
  left: 50%;
  top: 50%;
  border: 1px solid color-mix(in srgb, var(--stage-color) 38%, transparent);
  border-radius: 50%;
  transform: translate(-50%, -50%);
}
.o1 { width: 68%; height: 68%; animation: orbitSpin 7s linear infinite; }
.o2 { width: 46%; height: 46%; animation: orbitSpin 4.5s linear infinite reverse; }
.o3 { width: 26%; height: 26%; border-color: var(--stage-color); opacity: 0.45; }
.orbit i {
  position: absolute;
  top: 50%;
  left: -4px;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--stage-strong);
}
.o2 i { left: auto; right: -4px; background: var(--stage-strong); }

.cluster-core {
  position: absolute;
  color: var(--stage-color);
  font-size: 25%;
  animation: corePulse 1.6s ease-in-out infinite;
}

.planning {
  inset: 12% 20%;
}
.plan-root,
.plan-stem,
.plan-branch,
.plan-leaf {
  position: absolute;
  background: color-mix(in srgb, var(--stage-color) 55%, transparent);
}
.plan-root {
  left: 50%;
  top: 0;
  width: 2px;
  height: 20%;
  transform: translateX(-50%);
}
.plan-stem {
  left: 25%;
  right: 25%;
  top: 20%;
  height: 1px;
}
.plan-branch {
  top: 20%;
  width: 1px;
  height: 26%;
}
.p1 { left: 25%; }
.p2 { left: 50%; }
.p3 { left: 75%; }
.plan-leaf {
  width: 18%;
  height: 15%;
  top: 48%;
  border-radius: 4px;
  background: transparent;
  border: 1px solid color-mix(in srgb, var(--stage-color) 38%, transparent);
  animation: leafIn 2.2s ease-in-out infinite;
}
.l1 { left: 16%; }
.l2 { right: 16%; animation-delay: 0.2s; }

.sheets {
  inset: 18% 26%;
}
.sheet {
  position: absolute;
  inset: 0;
  border: 1px solid color-mix(in srgb, var(--stage-color) 45%, var(--border-color));
  border-radius: 7px;
  background: var(--card-bg);

}
.s1 { transform: translate(-22%, 16%) rotate(-7deg); opacity: 0.45; }
.s2 { transform: translate(-10%, 8%) rotate(-3deg); opacity: 0.7; }
.s3 { animation: pageFloat 3s ease-in-out infinite; }
.s4 {
  inset: 12% 10% 12% 18%;
  background: var(--stage-soft);
  border-color: transparent;
  animation: pageGlow 2.2s ease-in-out infinite;
}

.finalizing .final-ring {
  position: absolute;
  inset: 18%;
  border: 1px solid var(--border-color);
  border-top-color: var(--stage-strong);
  border-radius: 50%;
  animation: orbitSpin 1.1s linear infinite;
}
.finalizing .final-ring.inner {
  inset: 29%;
  border-top-color: var(--stage-color);
  animation-direction: reverse;
  animation-duration: 0.8s;
}
.final-core {
  position: absolute;
  color: var(--stage-color);
  font-size: 22%;
  animation: corePulse 1.1s ease-in-out infinite;
}

.complete-check {
  position: relative;
  width: 38%;
  height: 38%;
  border-radius: 50%;
  background: transparent;
  border: 1px solid var(--stage-color);
  color: var(--stage-strong);
  display: grid;
  place-items: center;
  font-size: calc(var(--size) * 0.14);
  animation: completeIn 0.6s ease-out both;

}

@keyframes nodePulse {
  0%, 100% { transform: scale(0.9); opacity: 0.55; }
  50% { transform: scale(1.18); opacity: 1; }
}
@keyframes edgePulse {
  0%, 100% { opacity: 0.25; }
  50% { opacity: 0.85; }
}
@keyframes scan {
  0%, 100% { top: 4%; opacity: 0.2; }
  50% { top: 94%; opacity: 1; }
}
@keyframes blockIn {
  0%, 100% { transform: translateY(2px); opacity: 0.55; }
  50% { transform: translateY(-2px); opacity: 1; }
}
@keyframes corePulse {
  0%, 100% { transform: scale(0.9); opacity: 0.5; }
  50% { transform: scale(1.12); opacity: 1; }
}
@keyframes orbitSpin {
  to { transform: translate(-50%, -50%) rotate(360deg); }
}
@keyframes leafIn {
  0%, 100% { opacity: 0.35; transform: translateY(3px); }
  50% { opacity: 1; transform: translateY(0); }
}
@keyframes pageFloat {
  0%, 100% { transform: translateY(1px); }
  50% { transform: translateY(-4px); }
}
@keyframes pageGlow {
  0%, 100% { opacity: 0.4; }
  50% { opacity: 0.95; }
}
@keyframes completeIn {
  from { transform: scale(0.65); opacity: 0; }
  to { transform: scale(1); opacity: 1; }
}
</style>
