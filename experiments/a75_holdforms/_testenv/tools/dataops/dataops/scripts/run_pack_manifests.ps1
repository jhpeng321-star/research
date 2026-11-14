# run_pack_manifests.ps1 -- generate all gpuA-pack manifests into manifests/ (background job driver)
$ErrorActionPreference = 'Continue'
$py = 'C:\Python314\python.exe'
$mk = 'E:\rd-wt\reorg3\dataops\scripts\make_pack_manifest.py'
$out = 'E:\rd-wt\reorg3\dataops\manifests'
$packs = @(
  @{n='pack_v3';              s='D:\rd-staging\wmn1_feat\pack_v3';                    b='/root/autodl-tmp/wmn1_m1_45c9032_20261003_A1/work/pack';        d='gdrive:research-archive/data/packs/pack_v3'},
  @{n='pack_dspeed';          s='E:\rd-wt\dspeed-data\_out\pack_dspeed';              b='/root/autodl-tmp/wmn1_a9_data/pack_dspeed';                     d='gdrive:research-archive/data/packs/pack_dspeed'},
  @{n='pack_damping';         s='E:\rd-wt\damping-snap\_out\pack_damping';            b='/root/autodl-tmp/wmn1_a10_data/pack_damping';                   d='gdrive:research-archive/data/packs/pack_damping'},
  @{n='identities_dspeed';    s='E:\rd-wt\a9\_out\identities_dspeed';                 b='/root/autodl-tmp/wms_run2_20261006/work/sidecars/identities_dspeed'; d='gdrive:research-archive/data/packs/sidecars/identities_dspeed'},
  @{n='identities_damping';   s='E:\rd-wt\damping-snap\_out\identities_damping';      b='/root/autodl-tmp/wmn1_a10_data/identities_damping';             d='gdrive:research-archive/data/packs/sidecars/identities_damping'},
  @{n='labels_overlay_dspeed'; s='E:\rd-wt\dspeed-data\_out\labels_overlay_dspeed';   b='/root/autodl-tmp/wmn1_a9_data/labels_overlay_dspeed';           d='gdrive:research-archive/data/packs/sidecars/labels_overlay_dspeed'},
  @{n='labels_overlay_damping'; s='E:\rd-wt\damping-snap\_out\labels_overlay_damping'; b='/root/autodl-tmp/wmn1_a10_data/labels_overlay_damping';        d='gdrive:research-archive/data/packs/sidecars/labels_overlay_damping'},
  @{n='sidecars_packv3_v2';   s='E:\rd-wt\ws-packv3fix\_out\sidecars_packv3_v2';      b='/root/autodl-tmp/wms_r3/sidecars_packv3_v2';                    d='gdrive:research-archive/data/packs/sidecars/sidecars_packv3_v2'}
)
$rc = 0
foreach ($p in $packs) {
  & $py $mk --pack $p.n --source $p.s --box $p.b --dest $p.d --out (Join-Path $out ($p.n + '.md5'))
  if ($LASTEXITCODE -ne 0) { $rc = $LASTEXITCODE }
}
Write-Output ("ALL_DONE rc=$rc")
