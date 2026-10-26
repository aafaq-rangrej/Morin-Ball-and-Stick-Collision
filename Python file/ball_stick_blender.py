"""
Ball-stick collisions (Morin 7.5) -> Blender.
Run headless:   blender -b -P ball_stick_blender.py -- --out ./ [--v0 1.0] [--h 0.4082] [--fps 60] [--no-follow]
Writes:  ball_stick_no_recollision.blend  and  ball_stick_with_recollision.blend
Each file has two scenes: "Elastic" (event-driven rigid-body sim, ported 1:1 from the HTML)
and "Inelastic" (ball sticks to stick end; analytic). Units: m = l = 1. Mapping: HTML (x,y) -> Blender (X,Z);
stick angle theta (clockwise +) = rotation_euler.y.
"""
import math, sys, os, argparse

I12 = 1/12; TP = 1.0; TMAX = 12.0
RB, RS = .05, .02; RR = RB + RS; DT = .001; STORE = 4

# ---------------- physics (pure python; identical logic to the HTML) ----------------
def sim_elastic(v0, h, recollide):
    st = dict(x=0, y=0, th=0, vx=0, vy=0, w=0, bx=-RR - v0*TP, by=h, bvx=v0, bvy=0)
    def adv(s, d):
        return dict(x=s['x']+s['vx']*d, y=s['y']+s['vy']*d, th=s['th']+s['w']*d, vx=s['vx'], vy=s['vy'], w=s['w'],
                    bx=s['bx']+s['bvx']*d, by=s['by']+s['bvy']*d, bvx=s['bvx'], bvy=s['bvy'])
    def contact(s):
        ux, uy = math.sin(s['th']), math.cos(s['th'])
        q = (s['bx']-s['x'])*ux + (s['by']-s['y'])*uy
        q = max(-.5, min(.5, q))
        px, py = s['x']+q*ux, s['y']+q*uy
        nx, ny = s['bx']-px, s['by']-py
        d = math.hypot(nx, ny) or 1e-12
        return d, nx/d, ny/d, px-s['x'], py-s['y']
    def vn(s, c):
        d, nx, ny, rx, ry = c
        return (s['bvx']-(s['vx']+s['w']*ry))*nx + (s['bvy']-(s['vy']-s['w']*rx))*ny
    def hit(s, c):
        d, nx, ny, rx, ry = c
        k = ry*nx - rx*ny
        j = -2*vn(s, c)/(2 + k*k/I12)
        s['bvx'] += j*nx; s['bvy'] += j*ny; s['vx'] -= j*nx; s['vy'] -= j*ny; s['w'] -= j*k/I12
    frames = [dict(st)]; colls = []; collP = []; t = 0.0; i = 0
    while t < TMAX:
        n1 = adv(st, DT); c = contact(n1)
        if (recollide or not colls) and c[0] < RR and vn(n1, c) < 0:
            lo, hi = 0.0, DT
            for _ in range(40):
                m = (lo+hi)/2
                if contact(adv(st, m))[0] < RR: hi = m
                else: lo = m
            st = adv(st, hi); cc = contact(st); hit(st, cc)
            colls.append(t+hi); collP.append((st['x']+cc[3], st['y']+cc[4]))
            st = adv(st, DT-hi)
        else:
            st = n1
        t += DT; i += 1
        if i % STORE == 0: frames.append(dict(st))
    return frames, colls, collP

def elastic_state(frames, t):
    f = frames[min(len(frames)-1, round(t/(DT*STORE)))]
    return dict(stick=(f['x'], f['y'], f['th']), ball=(f['bx'], f['by']),
                cm=((f['bx']+f['x'])/2, (f['by']+f['y'])/2))

def inelastic_state(v0, t):
    if t < TP:
        bx = -v0*(TP-t)
        return dict(stick=(0, 0, 0), ball=(bx, .5), cm=(bx/2, .25))
    s = t - TP; vc, w = v0/2, 6*v0/5
    px, py, th = vc*s, .25, w*s            # system CM
    ux, uy = math.sin(th), math.cos(th)
    return dict(stick=(px-.25*ux, py-.25*uy, th), ball=(px+.25*ux, py+.25*uy), cm=(px, py))

# ---------------- Blender scene building ----------------
def build(out_dir, v0, h, fps, follow):
    import bpy, bmesh
    from mathutils import Matrix
    bpy.context.preferences.edit.keyframe_new_interpolation_type = 'LINEAR'

    def mat(name, rgb):
        m = bpy.data.materials.new(name); m.diffuse_color = (*rgb, 1)
        return m
    M = {}
    def init_mats():  # must be re-created after every factory reset (reset deletes all datablocks)
        M['ball'] = mat('ball', (.76, .25, .05)); M['stick'] = mat('stick', (.11, .31, .85))
        M['cm'] = mat('cm', (.02, .47, .33)); M['col'] = mat('collision', (.76, .25, .05)); M['grid'] = mat('grid', (.7, .7, .68))

    def mesh_obj(scene, name, bm, material):
        me = bpy.data.meshes.new(name); bm.to_mesh(me); bm.free()
        me.materials.append(material)
        o = bpy.data.objects.new(name, me); scene.collection.objects.link(o); return o
    def sphere(r, at=(0, 0, 0)):
        bm = bmesh.new()
        bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=16, radius=r, matrix=Matrix.Translation(at))
        return bm
    def stick_bm():
        bm = bmesh.new()
        bmesh.ops.create_cone(bm, cap_ends=True, segments=24, radius1=RS, radius2=RS, depth=1.0)
        for z in (-.5, .5):
            bmesh.ops.create_uvsphere(bm, u_segments=24, v_segments=12, radius=RS, matrix=Matrix.Translation((0, 0, z)))
        return bm

    def make_scene(name, mode, recollide):
        sc = bpy.data.scenes.new(name)
        sc.render.fps = fps; sc.render.resolution_x = 960; sc.render.resolution_y = 400
        sc.frame_start = 0; sc.frame_end = int(TMAX*fps)
        sc.render.engine = 'BLENDER_WORKBENCH'
        sc.display.shading.light = 'FLAT'; sc.display.shading.color_type = 'MATERIAL'
        w = bpy.data.worlds.new(name+'_world'); w.color = (.98, .98, .97); sc.world = w
        # lab-frame reference dots (0.5 l spacing)
        bm = bmesh.new()
        for i in range(-8, 21):
            for j in range(-2, 4):
                bmesh.ops.create_cube(bm, size=.02, matrix=Matrix.Translation((i*.5, .3, j*.5)))
        mesh_obj(sc, 'grid', bm, M['grid'])
        stick = mesh_obj(sc, 'stick', stick_bm(), M['stick'])
        ball = mesh_obj(sc, 'ball', sphere(RB), M['ball'])
        cmo = mesh_obj(sc, 'system_CM', sphere(.03, (0, -.1, 0)), M['cm'])
        cam_d = bpy.data.cameras.new('cam'); cam_d.type = 'ORTHO'; cam_d.ortho_scale = 5.65
        cam = bpy.data.objects.new('camera', cam_d); sc.collection.objects.link(cam)
        cam.rotation_euler = (math.pi/2, 0, 0); sc.camera = cam

        if mode == 'e':
            frames, colls, collP = sim_elastic(v0, h, recollide)
            state = lambda t: elastic_state(frames, t)
        else:
            colls, collP = [TP], [(0, .5)]
            state = lambda t: inelastic_state(v0, t)
        for n in range(sc.frame_end + 1):
            s = state(n/fps)
            sx, sz, th = s['stick']
            stick.location = (sx, 0, sz); stick.rotation_euler = (0, th, 0)
            ball.location = (s['ball'][0], 0, s['ball'][1]); cmo.location = (s['cm'][0], 0, s['cm'][1])
            for o in (stick, ball, cmo):
                o.keyframe_insert('location', frame=n)
            stick.keyframe_insert('rotation_euler', frame=n)
            cam.location = (s['cm'][0] if follow else 0, -6, .3)
            cam.keyframe_insert('location', frame=n)
        # collision-event markers: appear (scale 0 -> 1) at the collision frame, stay at the lab-frame point
        for k, (tc, p) in enumerate(list(zip(colls, collP))[:60]):
            bm = bmesh.new()
            bmesh.ops.create_uvsphere(bm, u_segments=4, v_segments=2, radius=.035)  # octahedron-like diamond
            mk = mesh_obj(sc, f'collision_{k+1}', bm, M['col'])
            mk.location = (p[0], -.15, p[1])
            f = max(1, round(tc*fps))
            mk.scale = (0, 0, 0); mk.keyframe_insert('scale', frame=f-1)
            mk.scale = (1, 1, 1); mk.keyframe_insert('scale', frame=f)
        return sc

    for fname, rec in (('ball_stick_no_recollision', False), ('ball_stick_with_recollision', True)):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.context.preferences.edit.keyframe_new_interpolation_type = 'LINEAR'
        init_mats()
        make_scene('Elastic', 'e', rec)
        make_scene('Inelastic', 'i', rec)
        for sc in list(bpy.data.scenes):
            if sc.name not in ('Elastic', 'Inelastic'): bpy.data.scenes.remove(sc)
        path = os.path.join(out_dir, fname + '.blend')
        bpy.ops.wm.save_as_mainfile(filepath=path)
        print('wrote', path)

if __name__ == '__main__':
    argv = sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='.'); ap.add_argument('--v0', type=float, default=1.0)
    ap.add_argument('--h', type=float, default=1/math.sqrt(6)); ap.add_argument('--fps', type=int, default=60)
    ap.add_argument('--no-follow', action='store_true')
    a = ap.parse_args(argv)
    build(os.path.abspath(a.out), a.v0, a.h, a.fps, not a.no_follow)
