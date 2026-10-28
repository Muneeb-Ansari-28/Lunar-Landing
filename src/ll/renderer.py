"""Phase 6: High-Definition Pygame-CE Renderer for LunarLander-v3.

Renders realistic spacecraft, animated dual-layer supersonic plumes,
photorealistic lunar regolith with solar lighting, high-tech landing pad
with approach runway LEDs and telemetry beacons, dynamic ground spotlight
and dust ejecta physics, and a sleek aerospace glass-cockpit HUD.
"""
from __future__ import annotations
import math
import os
from pathlib import Path
import numpy as np
import pygame
from PIL import Image
from ll.ui_theme import (
    COLOR_SPACE_BG, COLOR_SPACE_STARS, COLOR_MOON_SURFACE,
    COLOR_MOON_SURFACE_DARK, COLOR_PAD, COLOR_PAD_MARK,
    COLOR_FLAG_POLE, COLOR_FLAG_CLOTH, COLOR_LANDER_BODY,
    COLOR_LANDER_WINDOW, COLOR_LANDER_LEG, COLOR_LANDER_FOOT,
    COLOR_FLAME_CORE, COLOR_FLAME_INNER, COLOR_FLAME_OUTER,
    COLOR_HUD_BG, COLOR_HUD_BORDER, COLOR_TEXT_TITLE,
    COLOR_TEXT_PRIMARY, COLOR_TEXT_MUTED, COLOR_TEXT_SUCCESS,
    COLOR_TEXT_WARNING, COLOR_TEXT_DANGER, CONFIG_COLORS,
    HUD_WIDTH, HUD_PADDING, LINE_HEIGHT
)

WORLD_W = 20.0  # 600 / 30
WORLD_H = 20.0  # Expanded vertical space (was 13.333) for high-altitude orbital descent

ACTION_NAMES = {
    0: "COASTING",
    1: "FIRE LEFT RCS",
    2: "MAIN THRUST 100%",
    3: "FIRE RIGHT RCS",
}


def _make_additive_spotlight(radius_x: int, radius_y: int, color: tuple[int, int, int]) -> pygame.Surface:
    """Precompute a smooth Gaussian radial light pool for ground illumination."""
    size_x, size_y = radius_x * 2, radius_y * 2
    y, x = np.ogrid[-radius_y:radius_y, -radius_x:radius_x]
    dist_sq = (x / radius_x) ** 2 + (y / radius_y) ** 2
    falloff = np.exp(-dist_sq * 3.2)
    falloff = np.where(dist_sq <= 1.0, falloff, 0.0)

    rgb = np.zeros((size_y, size_x, 3), dtype=np.uint8)
    rgb[:, :, 0] = np.clip(color[0] * falloff, 0, 255).astype(np.uint8)
    rgb[:, :, 1] = np.clip(color[1] * falloff, 0, 255).astype(np.uint8)
    rgb[:, :, 2] = np.clip(color[2] * falloff, 0, 255).astype(np.uint8)
    return pygame.image.frombuffer(rgb.tobytes(), (size_x, size_y), "RGB")


class LunarRenderer:
    """HD presentation renderer for LunarLander-v3."""

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        headless: bool = False,
        world_w: float = WORLD_W,
        world_h: float = WORLD_H,
    ):
        self.width = width
        self.height = height
        self.headless = headless
        self.world_w = world_w
        self.world_h = world_h

        if headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        else:
            os.environ.pop("SDL_VIDEODRIVER", None)

        pygame.init()
        pygame.font.init()

        if headless:
            self.screen = pygame.Surface((self.width, self.height))
        else:
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Lunar Lander Real-Time Physics & Landing Simulation")

        # Sleek Aerospace Fonts
        self.font_banner = pygame.font.SysFont("Impact", 26)
        self.font_hud_title = pygame.font.SysFont("Consolas", 15, bold=True)
        self.font_hud_bold = pygame.font.SysFont("Consolas", 13, bold=True)
        self.font_hud_regular = pygame.font.SysFont("Consolas", 12)
        self.font_hud_badge = pygame.font.SysFont("Consolas", 11, bold=True)
        self.font_controls = pygame.font.SysFont("Consolas", 12, bold=True)
        self.font_micro = pygame.font.SysFont("Consolas", 10, bold=True)

        # Procedural starfield (used if backdrop unavailable)
        rng = np.random.default_rng(1337)
        self.stars = []
        for _ in range(140):
            x = rng.integers(0, self.width)
            y = rng.integers(0, int(self.height * 0.75))
            brightness = rng.integers(140, 255)
            radius = rng.choice([1, 1, 2])
            self.stars.append((x, y, brightness, radius))

        self.flame_flicker = 0.0
        self._space_cache = None

        # --- Asset Pipeline ---
        asset_dir = Path(__file__).resolve().parents[2] / "assets"
        self.space_plate = self._load_space_backdrop(asset_dir / "space_backdrop.png")
        self.regolith_surf = self._load_regolith_texture(asset_dir)
        self.lander_art = self._load_cropped_alpha(asset_dir / "lander.png")
        self.engine_plume = self._load_cropped_alpha(asset_dir / "engine_plume.png")
        self.engine_core = self._load_cropped_alpha(asset_dir / "Code_Generated_Image.png")

        # Precomputed additive spotlights for ground engine illumination
        self.spotlight_gold = _make_additive_spotlight(140, 50, (255, 170, 50))
        self.spotlight_core = _make_additive_spotlight(70, 25, (255, 240, 200))

        # Terrain lighting cache to maintain 60+ FPS
        self._cached_terrain_pts: list[tuple[float, float]] | None = None
        self._cached_masked_terrain: pygame.Surface | None = None
        self._cached_screen_pts: list[tuple[int, int]] | None = None

    def _load_space_backdrop(self, path: Path) -> pygame.Surface | None:
        if not path.is_file():
            return None
        try:
            plate = pygame.image.load(str(path))
            return pygame.transform.smoothscale(plate, (self.width, self.height))
        except (pygame.error, OSError):
            return None

    def _load_regolith_texture(self, asset_dir: Path) -> pygame.Surface | None:
        # Prefer the high-contrast tiled regolith texture, fallback to lunar_regolith.png
        p1 = asset_dir / "Gemini_Generated_Image_cpnib4cpnib4cpni.jpg"
        p2 = asset_dir / "lunar_regolith.png"
        target_path = p1 if p1.is_file() else (p2 if p2.is_file() else None)
        if not target_path:
            return None
        try:
            with Image.open(str(target_path)) as im:
                scaled = im.resize((self.width, self.height), Image.Resampling.LANCZOS)
                return pygame.image.frombytes(scaled.tobytes(), (self.width, self.height), "RGB")
        except Exception:
            return None

    @staticmethod
    def _load_cropped_alpha(path: Path) -> pygame.Surface | None:
        if not path.is_file():
            return None
        try:
            surf = pygame.image.load(str(path))
            bbox = surf.get_bounding_rect(min_alpha=10)
            if bbox.width > 0 and bbox.height > 0:
                return surf.subsurface(bbox).copy()
            return surf
        except (pygame.error, OSError):
            return None

    def world_to_screen(self, wx: float, wy: float) -> tuple[int, int]:
        sx = int(wx * (self.width / self.world_w))
        sy = int(self.height - wy * (self.height / self.world_h))
        return sx, sy

    def render_frame(
        self,
        env,
        obs: np.ndarray,
        action: int = 0,
        config: str = "baseline",
        seed: int = 0,
        frame_idx: int = 0,
        raw_ret: float = 0.0,
        shaped_ret: float = 0.0,
        outcome: str | None = None,
        custom_hud_data: dict | None = None,
        paused: bool = False,
        speed: float = 1.0,
    ) -> np.ndarray:
        """Renders the scene and returns (H, W, 3) uint8 RGB array."""
        self.flame_flicker = (self.flame_flicker + 0.35) % (2 * math.pi)
        surf = self.screen

        # 1. Deep space vista & celestial bodies
        self._draw_space(surf, frame_idx)

        # 2. Extract physics geometry
        unwrapped = getattr(env, "unwrapped", env)
        pad_x1 = getattr(unwrapped, "helipad_x1", 8.0)
        pad_x2 = getattr(unwrapped, "helipad_x2", 12.0)
        pad_y = getattr(unwrapped, "helipad_y", 3.33333)

        # 3. Layered terrain and lighting
        self._draw_terrain(surf, unwrapped, pad_x1, pad_x2, pad_y)

        # 4. Precision landing platform & beacons
        self._draw_landing_pad(surf, pad_x1, pad_x2, pad_y, outcome, frame_idx)

        # 5. Spacecraft, propulsion plumes, and dynamic ground illumination
        self._draw_lander(surf, unwrapped, obs, action, frame_idx, pad_y, outcome)

        # 6. Sleek Glass-Cockpit Telemetry HUD
        self._draw_hud(
            surf,
            obs=obs,
            action=action,
            config=config,
            seed=seed,
            frame_idx=frame_idx,
            raw_ret=raw_ret,
            shaped_ret=shaped_ret,
            outcome=outcome,
            custom_hud_data=custom_hud_data,
            paused=paused,
            speed=speed,
        )

        # 7. Outcome Banner Overlay
        if outcome:
            self._draw_outcome_banner(surf, outcome, raw_ret)

        # 8. Bottom Controls Bar
        self._draw_controls_bar(surf)

        if not self.headless:
            pygame.display.flip()

        frame_data = pygame.surfarray.array3d(surf)
        return np.transpose(frame_data, (1, 0, 2))

    def _draw_space(self, surf: pygame.Surface, frame_idx: int):
        if self.space_plate:
            surf.blit(self.space_plate, (0, 0))
            return

        # Procedural fallback deep space
        if self._space_cache is None:
            self._draw_space_backdrop(surf, frame_idx)
            self._space_cache = surf.copy()
        else:
            surf.blit(self._space_cache, (0, 0))

        for sx, sy, brightness, radius in self.stars:
            pulse = 0.78 + 0.22 * math.sin(frame_idx * 0.06 + sx * 0.03 + sy * 0.02)
            b = max(65, min(255, int(brightness * pulse)))
            pygame.draw.circle(surf, (b, b, min(255, b + 18)), (sx, sy), radius)
            if radius > 1 and b > 205:
                pygame.draw.line(surf, (b, b, b), (sx - 3, sy), (sx + 3, sy), 1)
                pygame.draw.line(surf, (b, b, b), (sx, sy - 3), (sx, sy + 3), 1)

    def _draw_space_backdrop(self, surf: pygame.Surface, frame_idx: int):
        for y in range(self.height):
            t = y / max(1, self.height - 1)
            color = (int(5 + 8 * t), int(9 + 9 * t), int(20 + 14 * t))
            pygame.draw.line(surf, color, (0, y), (self.width, y))

        sun = (int(self.width * 0.17), int(self.height * 0.24))
        corona = pygame.Surface((88, 88), pygame.SRCALPHA)
        for radius, alpha in ((43, 8), (35, 13), (27, 23), (20, 48)):
            pygame.draw.circle(corona, (255, 175, 103, alpha), (44, 44), radius)
        surf.blit(corona, (sun[0] - 44, sun[1] - 44))
        pygame.draw.circle(surf, (255, 227, 174), sun, 17)
        pygame.draw.circle(surf, (255, 248, 225), (sun[0] - 3, sun[1] - 4), 10)

        # Distant worlds
        pygame.draw.circle(surf, (55, 76, 112), (int(self.width * 0.68), int(self.height * 0.20)), 28)
        pygame.draw.circle(surf, (119, 91, 74), (int(self.width * 0.58), int(self.height * 0.42)), 12)

    def _draw_terrain(self, surf: pygame.Surface, unwrapped, pad_x1: float, pad_x2: float, pad_y: float):
        # Extract Box2D surface contour points
        moon = getattr(unwrapped, "moon", None)
        contour_pts = []
        if moon and hasattr(moon, "fixtures") and len(moon.fixtures) > 0:
            top_pts = []
            for f in moon.fixtures:
                shape = f.shape
                if hasattr(shape, "vertices") and len(shape.vertices) == 2:
                    v1, v2 = shape.vertices
                    if v1[1] > 0.01 or v2[1] > 0.01:
                        top_pts.append((float(v1[0]), float(v1[1])))
                        top_pts.append((float(v2[0]), float(v2[1])))
            if top_pts:
                # Remove duplicate vertices and sort left-to-right
                unique_pts = list({(round(p[0], 4), round(p[1], 4)): p for p in top_pts}.values())
                contour_pts = sorted(unique_pts, key=lambda p: p[0])

        if not contour_pts:
            num_segments = 25
            for i in range(num_segments + 1):
                wx = (i / num_segments) * WORLD_W
                if pad_x1 - 0.2 <= wx <= pad_x2 + 0.2:
                    wy = pad_y
                else:
                    wy = pad_y - 0.6 + 0.4 * math.sin(wx * 0.9) + 0.2 * math.sin(wx * 2.5)
                contour_pts.append((wx, wy))

        # Check if terrain can be reused from cache
        if self._cached_terrain_pts != contour_pts or self._cached_masked_terrain is None:
            self._cached_terrain_pts = contour_pts
            screen_pts = [self.world_to_screen(wx, wy) for wx, wy in contour_pts]
            self._cached_screen_pts = screen_pts
            ground_poly = [(0, self.height)] + screen_pts + [(self.width, self.height)]

            terrain_mask = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            pygame.draw.polygon(terrain_mask, (255, 255, 255, 255), ground_poly)

            if self.regolith_surf is not None:
                textured_ground = self.regolith_surf.copy()
            else:
                textured_ground = pygame.Surface((self.width, self.height))
                textured_ground.fill((120, 125, 138))

            # Solar lighting modulation: Sun is at top-left (150, 120)
            sun_x, sun_y = 150.0, 120.0
            light_overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            for i in range(len(screen_pts) - 1):
                p1 = screen_pts[i]
                p2 = screen_pts[i + 1]
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                seg_len = math.hypot(dx, dy)
                if seg_len < 1e-4:
                    continue
                nx = -dy / seg_len
                ny = dx / seg_len
                mx = (p1[0] + p2[0]) / 2.0
                my = (p1[1] + p2[1]) / 2.0
                lx = sun_x - mx
                ly = sun_y - my
                l_len = math.hypot(lx, ly)
                lx /= l_len
                ly /= l_len
                dot = nx * lx + ny * ly
                poly_strip = [(p1[0], p1[1]), (p2[0], p2[1]), (p2[0], self.height), (p1[0], self.height)]
                if dot > 0.15:
                    alpha = int(min(65, (dot - 0.15) * 75))
                    pygame.draw.polygon(light_overlay, (255, 245, 220, alpha), poly_strip)
                else:
                    alpha = int(min(70, (0.15 - dot) * 80))
                    pygame.draw.polygon(light_overlay, (15, 20, 30, alpha), poly_strip)

            # Subsurface depth shading
            depth_overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            start_y = int(self.height * 0.5)
            for y_coord in range(start_y, self.height):
                factor = (y_coord - start_y) / (self.height - start_y)
                alpha = int(factor * 55)
                pygame.draw.line(depth_overlay, (10, 12, 18, alpha), (0, y_coord), (self.width, y_coord))

            textured_ground.blit(light_overlay, (0, 0))
            textured_ground.blit(depth_overlay, (0, 0))

            # Apply alpha mask
            masked = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            masked.blit(textured_ground, (0, 0))
            alpha_mask = pygame.surfarray.pixels_alpha(terrain_mask)
            target_alpha = pygame.surfarray.pixels_alpha(masked)
            target_alpha[:] = alpha_mask
            del alpha_mask
            del target_alpha

            self._cached_masked_terrain = masked

        surf.blit(self._cached_masked_terrain, (0, 0))

        # Sunlit crest highlight line & contact shadow
        if self._cached_screen_pts:
            pygame.draw.lines(surf, (30, 35, 45), False, [(x, y + 2) for x, y in self._cached_screen_pts], 2)
            pygame.draw.lines(surf, (220, 230, 245), False, self._cached_screen_pts, 2)

    def _draw_landing_pad(
        self,
        surf: pygame.Surface,
        pad_x1: float,
        pad_x2: float,
        pad_y: float,
        outcome: str | None,
        frame_idx: int,
    ):
        sx1, sy1 = self.world_to_screen(pad_x1, pad_y)
        sx2, sy2 = self.world_to_screen(pad_x2, pad_y)
        pad_w = sx2 - sx1
        pad_h = 14

        # 1. Beveled foundation base
        base_poly = [
            (sx1 - 8, sy1 + pad_h + 2),
            (sx1 - 3, sy1 + 2),
            (sx2 + 3, sy1 + 2),
            (sx2 + 8, sy1 + pad_h + 2),
        ]
        pygame.draw.polygon(surf, (22, 26, 36), base_poly)
        pygame.draw.polygon(surf, (45, 55, 72), base_poly, 1)

        # 2. Machined carbon/titanium landing slab
        deck_rect = pygame.Rect(sx1, sy1, pad_w, pad_h)
        pygame.draw.rect(surf, (34, 40, 52), deck_rect, border_radius=2)
        pygame.draw.line(surf, (150, 175, 200), (sx1, sy1), (sx2, sy1), 2)  # Sunlit bevel

        # 3. Precision hazard chevron border
        stripe_w = 12
        for x in range(sx1 + 4, sx2 - stripe_w, stripe_w * 2):
            pts = [(x, sy1 + pad_h - 4), (x + 6, sy1 + 4), (x + 12, sy1 + 4), (x + 6, sy1 + pad_h - 4)]
            pygame.draw.polygon(surf, (220, 160, 35), pts)

        # 4. Center Touchdown Target [ + ]
        cx_pad = (sx1 + sx2) // 2
        cy_pad = sy1 + pad_h // 2
        target_color = (60, 255, 120) if outcome == "landed" else (0, 225, 255)
        pygame.draw.circle(surf, target_color, (cx_pad, cy_pad), 16, 2)
        pygame.draw.circle(surf, (240, 245, 255), (cx_pad, cy_pad), 5, 1)
        pygame.draw.line(surf, target_color, (cx_pad - 24, cy_pad), (cx_pad - 10, cy_pad), 2)
        pygame.draw.line(surf, target_color, (cx_pad + 10, cy_pad), (cx_pad + 24, cy_pad), 2)
        pygame.draw.line(surf, target_color, (cx_pad, cy_pad - 10), (cx_pad, cy_pad - 5), 2)
        pygame.draw.line(surf, target_color, (cx_pad, cy_pad + 5), (cx_pad, cy_pad + 10), 2)

        # 5. Runway Approach LED Light Nodes
        num_leds = 14
        if outcome == "landed":
            led_color = (60, 255, 120)
            led_halo = (40, 230, 100, 80)
        elif outcome == "crashed":
            led_color = (255, 50, 50)
            led_halo = (255, 40, 40, 80)
        else:
            led_color = (220, 250, 255)
            led_halo = (0, 220, 255, 60)

        for i in range(num_leds):
            lx = sx1 + int((i + 0.5) * (pad_w / num_leds))
            ly = sy1 + 1
            halo = pygame.Surface((12, 12), pygame.SRCALPHA)
            pygame.draw.circle(halo, led_halo, (6, 6), 5)
            pygame.draw.circle(halo, led_color, (6, 6), 2)
            surf.blit(halo, (lx - 6, ly - 6), special_flags=pygame.BLEND_ADD)

        # Pad Label
        lz_lbl = self.font_micro.render("LZ-1 ALPHA", True, target_color)
        surf.blit(lz_lbl, (cx_pad - 26, sy1 + pad_h + 3))

        # 6. Telemetry Flag Masts & Beacons
        beacon_color = (60, 255, 120) if outcome == "landed" else (255, 185, 45)
        for side, fx in [("left", sx1), ("right", sx2)]:
            pygame.draw.line(surf, (20, 24, 32), (fx + 1, sy1), (fx + 1, sy1 - 42), 3)
            pygame.draw.line(surf, (200, 215, 235), (fx, sy1), (fx, sy1 - 42), 2)
            beacon = pygame.Surface((24, 24), pygame.SRCALPHA)
            pulse = 0.8 + 0.2 * math.sin(frame_idx * 0.25)
            pygame.draw.circle(beacon, (*beacon_color, int(90 * pulse)), (12, 12), 10)
            pygame.draw.circle(beacon, (255, 245, 210, 255), (12, 12), 3)
            surf.blit(beacon, (fx - 12, sy1 - 52), special_flags=pygame.BLEND_ADD)

            # Pennant waving flag
            wave = 2.0 * math.sin(frame_idx * 0.12 + (0 if side == "left" else 1.5))
            if side == "left":
                pennant = [(fx, sy1 - 42), (fx + 24, int(sy1 - 34 + wave)), (fx, sy1 - 24)]
            else:
                pennant = [(fx, sy1 - 42), (fx - 24, int(sy1 - 34 + wave)), (fx, sy1 - 24)]
            pygame.draw.polygon(surf, (220, 48, 48), pennant)
            pygame.draw.polygon(surf, (255, 215, 70), pennant, 1)

    def _draw_lander(
        self,
        surf: pygame.Surface,
        unwrapped,
        obs: np.ndarray,
        action: int,
        frame_idx: int,
        pad_y: float,
        outcome: str | None,
    ):
        lander = getattr(unwrapped, "lander", None)
        if lander:
            wx, wy = lander.position[0], lander.position[1]
            angle = lander.angle
        else:
            wx = obs[0] * 10.0 + 10.0
            wy = obs[1] * 6.666667 + 3.333333
            angle = obs[4]

        cx, cy = self.world_to_screen(wx, wy)

        if self.lander_art is None:
            self._draw_procedural_lander(surf, cx, cy, angle, action)
            return

        ship_h = max(68, int(self.height * 0.15))  # ~108px at 720p
        ship_w = int(ship_h * self.lander_art.get_width() / self.lander_art.get_height())
        ship = pygame.transform.smoothscale(self.lander_art, (ship_w, ship_h))

        layer_size = max(340, int(ship_h * 3.6))
        layer = pygame.Surface((layer_size, layer_size), pygame.SRCALPHA)
        origin = layer_size // 2

        # In lander.png bounding box, nozzle bell exit is at +0.383 * ship_h
        nozzle_x = origin
        nozzle_y = origin + int(0.383 * ship_h)

        # Distance from nozzle to ground in screen pixels
        pad_screen_y = self.world_to_screen(0.0, pad_y)[1]
        dist_to_ground = max(0, pad_screen_y - (cy + int(0.383 * ship_h)))

        # 1. Main Propulsion Plume (Action == 2)
        if action == 2 and self.engine_plume and self.engine_core and outcome != "crashed":
            nominal_plume_h = int(ship_h * (1.55 + 0.10 * math.sin(self.flame_flicker)))
            plume_h = max(24, min(nominal_plume_h, dist_to_ground + 10))
            plume_w = int(ship_w * 0.44 * (1.0 + 0.05 * math.sin(self.flame_flicker * 1.8)))

            plume_scaled = pygame.transform.smoothscale(self.engine_plume, (plume_w, plume_h))
            core_scaled = pygame.transform.smoothscale(
                self.engine_core, (int(plume_w * 0.75), int(plume_h * 0.95))
            )

            p_rect = plume_scaled.get_rect(midtop=(nozzle_x, nozzle_y - 4))
            c_rect = core_scaled.get_rect(midtop=(nozzle_x, nozzle_y - 4))

            # Composite wispy plume then additive Mach core
            layer.blit(plume_scaled, p_rect)
            layer.blit(core_scaled, c_rect, special_flags=pygame.BLEND_RGBA_ADD)

            # High-velocity incandescent exhaust sparks
            rng = np.random.default_rng(frame_idx + 1337)
            for _ in range(16):
                t = rng.uniform(0.1, 1.0)
                spark_x = nozzle_x + int(rng.normal(0, 3 + t * 10))
                spark_y = nozzle_y + int(t * plume_h)
                sz = rng.choice([1, 1, 2])
                alpha = int(240 * (1.0 - t * 0.8))
                dot = pygame.Surface((sz * 2, sz * 2), pygame.SRCALPHA)
                pygame.draw.circle(dot, (255, 230, 160, alpha), (sz, sz), sz)
                layer.blit(dot, (spark_x - sz, spark_y - sz), special_flags=pygame.BLEND_RGBA_ADD)

        # 2. Side RCS Vernier Thrusters (Actions 1 & 3)
        elif action in (1, 3) and outcome != "crashed":
            side_sign = -1 if action == 1 else 1
            rcs_x = origin + int(side_sign * 0.29 * ship_w)
            rcs_y = origin - int(0.10 * ship_h)
            jet_len = 36
            jet_w = 14

            jet_pts = [
                (rcs_x, rcs_y),
                (rcs_x + side_sign * jet_len, rcs_y - jet_w // 2),
                (rcs_x + side_sign * (jet_len + 6), rcs_y),
                (rcs_x + side_sign * jet_len, rcs_y + jet_w // 2),
            ]
            pygame.draw.polygon(layer, (140, 230, 255, 180), jet_pts)
            inner_core = [
                (rcs_x, rcs_y),
                (rcs_x + side_sign * int(jet_len * 0.6), rcs_y - 3),
                (rcs_x + side_sign * int(jet_len * 0.65), rcs_y),
                (rcs_x + side_sign * int(jet_len * 0.6), rcs_y + 3),
            ]
            pygame.draw.polygon(layer, (255, 255, 255, 240), inner_core)

            rng = np.random.default_rng(frame_idx + 888)
            for _ in range(10):
                t = rng.uniform(0.2, 1.2)
                px = rcs_x + side_sign * int(t * jet_len)
                py = rcs_y + int(rng.normal(0, 1 + t * 3.5))
                sz = rng.choice([1, 1, 2])
                alpha = int(max(0, 220 * (1.0 - t / 1.3)))
                dot = pygame.Surface((sz * 2, sz * 2), pygame.SRCALPHA)
                pygame.draw.circle(dot, (180, 240, 255, alpha), (sz, sz), sz)
                layer.blit(dot, (px - sz, py - sz))

        # 3. Blit lander on top of plume so nozzle bell overlaps the flame top!
        ship_rect = ship.get_rect(center=(origin, origin))
        layer.blit(ship, ship_rect)

        # 4. Rotate and blit to world screen
        angle_deg = -math.degrees(angle)
        rotated = pygame.transform.rotozoom(layer, angle_deg, 1.0)
        surf.blit(rotated, rotated.get_rect(center=(cx, cy)))

        # 5. Dynamic Ground Illumination & Ballistic Dust Ejection
        alt_norm = max(0.0, min(1.0, (wy - pad_y) / 2.5))
        if action == 2 and alt_norm < 0.8 and outcome != "crashed":
            intensity = (1.0 - alt_norm / 0.8) ** 1.5
            surf.blit(
                self.spotlight_gold,
                (cx - 140, pad_screen_y - 25),
                special_flags=pygame.BLEND_ADD,
            )
            surf.blit(
                self.spotlight_core,
                (cx - 70, pad_screen_y - 12),
                special_flags=pygame.BLEND_ADD,
            )

            # High-speed ballistic surface dust ejecta
            rng = np.random.default_rng(frame_idx + 77)
            max_dist = max(5.0, 170.0 * intensity)
            for _ in range(45):
                side = rng.choice([-1, 1])
                dist = rng.uniform(0.0, max_dist)
                dust_x = int(cx + side * dist)
                dust_y = int(pad_screen_y + rng.uniform(-4, 3))
                alpha = int(max(0, 210 * (1.0 - dist / 180)))
                sz = rng.choice([1, 1, 2])
                dust_dot = pygame.Surface((sz * 2, sz * 2), pygame.SRCALPHA)
                pygame.draw.circle(dust_dot, (235, 220, 190, alpha), (sz, sz), sz)
                surf.blit(dust_dot, (dust_x, dust_y), special_flags=pygame.BLEND_ADD)

        # 6. Crash Impact Flash & Debris
        if outcome == "crashed":
            crash_flash = pygame.Surface((180, 180), pygame.SRCALPHA)
            pygame.draw.circle(crash_flash, (255, 80, 20, 130), (90, 90), 80)
            pygame.draw.circle(crash_flash, (255, 200, 60, 220), (90, 90), 40)
            surf.blit(crash_flash, (cx - 90, cy - 90), special_flags=pygame.BLEND_ADD)

    def _draw_procedural_lander(self, surf, cx, cy, angle, action):
        """High-detail vector fallback if lander image asset is missing."""
        bw, bh = 42, 50
        ship_surf = pygame.Surface((bw * 3, bh * 3), pygame.SRCALPHA)
        ship_cx, ship_cy = int(bw * 1.5), int(bh * 1.5)

        if action == 2:
            flame_y = ship_cy + int(bh * 0.44)
            flame_len = int(48 + 12 * math.sin(self.flame_flicker))
            pygame.draw.polygon(
                ship_surf,
                (255, 87, 27),
                [(ship_cx - 12, flame_y), (ship_cx, flame_y + flame_len), (ship_cx + 12, flame_y)],
            )
            pygame.draw.polygon(
                ship_surf,
                (255, 242, 190),
                [(ship_cx - 4, flame_y), (ship_cx, flame_y + int(flame_len * 0.45)), (ship_cx + 4, flame_y)],
            )

        # Legs
        for side in (-1, 1):
            shoulder = (ship_cx + side * 12, ship_cy + 13)
            knee = (ship_cx + side * 21, ship_cy + 35)
            foot = (ship_cx + side * 29, ship_cy + 39)
            pygame.draw.line(ship_surf, (193, 199, 207), shoulder, knee, 3)
            pygame.draw.line(ship_surf, (139, 149, 165), knee, foot, 3)
            pygame.draw.line(ship_surf, (244, 202, 113), (foot[0] - 5, foot[1]), (foot[0] + 5, foot[1]), 4)

        # Hull & Cockpit
        hull = [
            (ship_cx - 12, ship_cy - 12),
            (ship_cx - 8, ship_cy - 23),
            (ship_cx + 8, ship_cy - 23),
            (ship_cx + 12, ship_cy - 12),
            (ship_cx + 10, ship_cy + 13),
            (ship_cx - 10, ship_cy + 13),
        ]
        pygame.draw.polygon(ship_surf, (181, 193, 209), hull)
        pygame.draw.polygon(ship_surf, (226, 231, 237), hull, 2)
        pygame.draw.circle(ship_surf, (44, 122, 178), (ship_cx, ship_cy - 8), 7)

        rotated = pygame.transform.rotozoom(ship_surf, -math.degrees(angle), 1.0)
        surf.blit(rotated, rotated.get_rect(center=(cx, cy)))

    def _draw_hud(
        self,
        surf: pygame.Surface,
        obs: np.ndarray,
        action: int,
        config: str,
        seed: int,
        frame_idx: int,
        raw_ret: float,
        shaped_ret: float,
        outcome: str | None,
        custom_hud_data: dict | None,
        paused: bool,
        speed: float,
    ):
        hud_w = 310
        hud_h = 330
        hud_x = self.width - hud_w - 20
        hud_y = 18

        hud_surf = pygame.Surface((hud_w, hud_h), pygame.SRCALPHA)
        hud_surf.fill((8, 14, 24, 170))  # Polarized translucent glass
        pygame.draw.rect(hud_surf, (35, 75, 120, 180), (0, 0, hud_w, hud_h), width=1, border_radius=8)

        # High-tech cyber corner brackets
        b_len = 10
        b_col = (0, 220, 255)
        for bx, by, dx, dy in [
            (0, 0, 1, 1),
            (hud_w - 1, 0, -1, 1),
            (0, hud_h - 1, 1, -1),
            (hud_w - 1, hud_h - 1, -1, -1),
        ]:
            pygame.draw.line(hud_surf, b_col, (bx, by), (bx + dx * b_len, by), 2)
            pygame.draw.line(hud_surf, b_col, (bx, by), (bx, by + dy * b_len), 2)

        # Header Title & Status Dot
        title = self.font_hud_title.render("APOLLO TELEMETRY", True, (240, 245, 255))
        hud_surf.blit(title, (14, 12))

        dot_col = (50, 230, 100) if not paused else (255, 180, 40)
        pygame.draw.circle(hud_surf, dot_col, (hud_w - 75, 20), 4)
        status_txt = self.font_hud_badge.render("PAUSED" if paused else "ONLINE", True, dot_col)
        hud_surf.blit(status_txt, (hud_w - 65, 14))

        # Thin divider
        pygame.draw.line(hud_surf, (35, 75, 120, 150), (12, 36), (hud_w - 12, 36), 1)

        # Propulsion status styling
        if action == 2:
            prop_txt = "[ MAIN THRUST 100% ]"
            prop_col = (255, 175, 40)
        elif action == 1:
            prop_txt = "[ RCS PORT ]"
            prop_col = (0, 230, 255)
        elif action == 3:
            prop_txt = "[ RCS STBD ]"
            prop_col = (0, 230, 255)
        else:
            prop_txt = "[ COASTING ]"
            prop_col = (130, 150, 180)

        # Vertical velocity safe indicator
        vy = obs[3]
        if vy > -0.25:
            vy_tag = "[SAFE]"
            vy_col = (60, 230, 120)
        elif vy > -0.55:
            vy_tag = "[WARN]"
            vy_col = (255, 190, 40)
        else:
            vy_tag = "[DANGER]"
            vy_col = (255, 70, 70)

        # Tilt angle in degrees
        tilt_deg = math.degrees(obs[4])
        tilt_tag = "[LEVEL]" if abs(tilt_deg) <= 12.0 else "[TILT]"

        # Landing Gear Sensors
        l_contact, r_contact = bool(obs[6]), bool(obs[7])
        gear_txt = f"L: {'LOCKED' if l_contact else 'READY'}  R: {'LOCKED' if r_contact else 'READY'}"
        gear_col = (60, 230, 120) if (l_contact or r_contact) else (120, 180, 240)

        cfg_color = CONFIG_COLORS.get(config, (60, 150, 255))
        pause_tag = " [PAUSED]" if paused else f" [{speed:.1f}x]"
        alt_mode = (custom_hud_data or {}).get("altitude_mode", "high")
        alt_tag = "HIGH ORBIT" if alt_mode == "high" else "STANDARD"

        rows = [
            ("FLIGHT MODE", f"{config.upper()}{pause_tag}", cfg_color),
            ("ORBIT PROFILE", alt_tag, (0, 220, 255) if alt_mode == "high" else (170, 190, 215)),
            ("PROPULSION", prop_txt, prop_col),
            ("ALTITUDE (y)", f"{obs[1]:+.3f} m", (100, 240, 180) if obs[1] < 0.5 else (210, 225, 240)),
            ("DESCENT (Vy)", f"{vy:+.3f} m/s {vy_tag}", vy_col),
            ("DRIFT (Vx)", f"{obs[2]:+.3f} m/s", (210, 225, 240)),
            ("ATTITUDE (θ)", f"{tilt_deg:+.1f}° {tilt_tag}", (210, 225, 240)),
            ("GEAR SENSORS", gear_txt, gear_col),
            ("RAW SCORE", f"{raw_ret:+.2f} PTS", (60, 230, 120) if raw_ret >= 100 else (210, 225, 240)),
            ("MISSION TIME", f"{frame_idx / 50.0:.2f}s (F{frame_idx:04d})", (180, 195, 215)),
        ]

        if outcome:
            out_col = (60, 230, 120) if outcome == "landed" else (255, 70, 70)
            rows.append(("STATUS", outcome.upper(), out_col))

        y_row = 44
        for label, val, color in rows[:10]:
            lbl_s = self.font_hud_regular.render(label, True, (130, 150, 180))
            val_s = self.font_hud_bold.render(val, True, color)
            hud_surf.blit(lbl_s, (14, y_row))
            hud_surf.blit(val_s, (136, y_row))
            y_row += 21

        # Descent Rate Safety Window Gauge
        bar_y = y_row + 4
        bar_w = hud_w - 28
        pygame.draw.rect(hud_surf, (20, 30, 45), (14, bar_y, bar_w, 10), border_radius=3)
        # Safe landing zone window
        pygame.draw.rect(hud_surf, (40, 160, 80), (14 + bar_w // 2 - 28, bar_y, 56, 10), border_radius=2)
        # Dynamic cursor representing current Vy
        norm_vy = np.clip(vy, -1.0, 1.0)
        cursor_x = int(14 + bar_w // 2 + norm_vy * (bar_w // 2 - 10))
        pygame.draw.line(hud_surf, (255, 255, 255), (cursor_x, bar_y - 2), (cursor_x, bar_y + 12), 2)

        bar_lbl = self.font_hud_badge.render("DESCENT RATE SAFE WINDOW", True, (120, 150, 180))
        hud_surf.blit(bar_lbl, (14, bar_y + 14))

        surf.blit(hud_surf, (hud_x, hud_y))

    def _draw_outcome_banner(self, surf: pygame.Surface, outcome: str, raw_ret: float):
        if outcome == "landed":
            text = f"MISSION SUCCESS: SAFE TOUCHDOWN ({raw_ret:+.1f} PTS)"
            bg_col = (12, 45, 25, 220)
            border_col = (50, 230, 110)
            txt_col = (210, 255, 220)
        elif outcome == "crashed":
            text = f"CRITICAL FAILURE: LANDER CRASHED ({raw_ret:+.1f} PTS)"
            bg_col = (55, 16, 18, 220)
            border_col = (255, 60, 60)
            txt_col = (255, 215, 215)
        else:
            text = f"MISSION ABORT: FUEL/TIME EXHAUSTED ({raw_ret:+.1f} PTS)"
            bg_col = (50, 35, 12, 220)
            border_col = (255, 180, 40)
            txt_col = (255, 240, 205)

        banner_surf = self.font_banner.render(text, True, txt_col)
        bw, bh = banner_surf.get_width() + 40, banner_surf.get_height() + 16
        # Center in the flight arena to the left of the telemetry HUD
        flight_arena_w = self.width - 340
        bx = (flight_arena_w - bw) // 2
        by = int(self.height * 0.18)

        card = pygame.Surface((bw, bh), pygame.SRCALPHA)
        card.fill(bg_col)
        pygame.draw.rect(card, border_col, card.get_rect(), width=2, border_radius=8)
        card.blit(banner_surf, (20, 8))
        surf.blit(card, (bx, by))

    def _draw_controls_bar(self, surf: pygame.Surface):
        bar_rect = pygame.Rect(20, self.height - 36, self.width - 40, 28)
        bar_surf = pygame.Surface((bar_rect.width, bar_rect.height), pygame.SRCALPHA)
        bar_surf.fill((8, 12, 20, 190))
        pygame.draw.rect(bar_surf, (35, 65, 100), bar_surf.get_rect(), width=1, border_radius=6)

        hint = self.font_controls.render(
            "CONTROLS: [SPACE] Pause  |  [R] Restart  |  [A] Altitude (High/Std)  |  [C] Policy  |  [1-4] Speed  |  [ESC/Q] Exit",
            True,
            (180, 205, 235),
        )
        bar_surf.blit(hint, (15, 6))
        surf.blit(bar_surf, bar_rect.topleft)

    def close(self):
        pygame.quit()
