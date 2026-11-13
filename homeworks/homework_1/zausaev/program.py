"""Домашняя работа № 1 по многофазным потокам.

Выполнил: Владимир Заусаев
Вариант: 4

Расчёт VLP нагнетательной скважины для однофазного потока воды.
"""

import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp


ATM_TO_PA = 101325.0
SECONDS_PER_DAY = 24 * 60 * 60


def calc_ws(gamma_wat: float) -> float:
    """Рассчитать массовое солесодержание воды, г/г."""
    rho_sc = gamma_wat * 1000
    ws = (
        1
        / rho_sc
        * (1.36545 * rho_sc - (3838.77 * rho_sc - 2.009 * rho_sc**2) ** 0.5)
    )
    return max(ws, 0)


def calc_rho_w(ws: float, t: float) -> float:
    """Рассчитать плотность воды, кг/м³; температура ``t`` задана в K."""
    rho_w = 1000 * (1.0009 - 0.7114 * ws + 0.2605 * ws**2) ** (-1)
    return rho_w / (
        1 + (t - 273) * 1e-4 * (0.269 * (t - 273) ** 0.637 - 0.8)
    )


def calc_mu_w(ws: float, t: float, p: float) -> float:
    """Рассчитать динамическую вязкость воды, сП, по Matthews & Russell."""
    a = (
        109.574
        - 0.840564 * 1000 * ws
        + 3.13314 * 1000 * ws**2
        + 8.72213 * 1000 * ws**3
    )
    b = (
        1.12166
        - 2.63951 * ws
        + 6.79461 * ws**2
        + 54.7119 * ws**3
        - 155.586 * ws**4
    )
    p_mpa = p * 1e-6
    return (
        a
        * (1.8 * t - 460) ** (-b)
        * (0.9994 + 0.0058 * p_mpa + 0.6534e-4 * p_mpa**2)
    )


def calc_n_re(rho_w: float, q_ms: float, mu_w: float, d_tub: float) -> float:
    """Рассчитать число Рейнольдса."""
    velocity = q_ms / (np.pi * d_tub**2 / 4)
    return rho_w * velocity * d_tub / mu_w * 1000


def calc_ff_churchill(n_re: float, roughness: float, d_tub: float) -> float:
    """Рассчитать коэффициент трения по корреляции Churchill."""
    a = (-2.457 * np.log((7 / n_re) ** 0.9 + 0.27 * roughness / d_tub)) ** 16
    b = (37530 / n_re) ** 16
    return 8 * ((8 / n_re) ** 12 + 1 / (a + b) ** 1.5) ** (1 / 12)


def calc_dp_dl_grav(rho_w: float, angle: float) -> float:
    """Рассчитать гидростатический градиент, Па/м."""
    return rho_w * 9.81 * np.sin(np.radians(angle))


def calc_dp_dl_fric(
    rho_w: float,
    mu_w: float,
    q_ms: float,
    d_tub: float,
    roughness: float,
) -> float:
    """Рассчитать составляющую градиента давления на трение, Па/м."""
    if q_ms == 0:
        return 0.0
    n_re = calc_n_re(rho_w, q_ms, mu_w, d_tub)
    ff = calc_ff_churchill(n_re, roughness, d_tub)
    return ff * rho_w * q_ms**2 / d_tub**5


def calc_dp_dl(
    rho_w: float,
    mu_w: float,
    angle: float,
    q_ms: float,
    d_tub: float,
    roughness: float,
) -> float:
    """Рассчитать полный градиент давления при движении воды вниз, Па/м."""
    dp_dl_grav = calc_dp_dl_grav(rho_w, angle)
    dp_dl_fric = calc_dp_dl_fric(rho_w, mu_w, q_ms, d_tub, roughness)
    return dp_dl_grav - 0.815 * dp_dl_fric


def _integr_func(
    h: float,
    pt: tuple,
    temp_grad: float,
    gamma_wat: float,
    angle: float,
    q_ms: float,
    d_tub: float,
    roughness: float,
) -> tuple:
    """Вернуть градиенты давления и температуры для интегрирования."""
    del h
    p, t = pt
    ws = calc_ws(gamma_wat)
    rho_w = calc_rho_w(ws, t)
    mu_w = calc_mu_w(ws, t, p)
    dp_dl = calc_dp_dl(rho_w, mu_w, angle, q_ms, d_tub, roughness)
    dt_dl = temp_grad / 100
    return dp_dl, dt_dl


def calc_pipe(
    p_wh: float,
    t_wh: float,
    h0: float,
    md_vdp: float,
    temp_grad: float,
    gamma_wat: float,
    angle: float,
    q_ms: float,
    d_tub: float,
    roughness: float,
) -> tuple:
    """Рассчитать распределение давления и температуры вдоль НКТ."""
    result = solve_ivp(
        _integr_func,
        (h0, md_vdp),
        (p_wh, t_wh),
        args=(temp_grad, gamma_wat, angle, q_ms, d_tub, roughness),
        method="RK45",
        max_step=5,
    )
    if not result.success:
        raise RuntimeError(result.message)
    return result.y[0], result.y[1], result.t


def calc_p_wf(
    p_wh: float,
    t_wh: float,
    h0: float,
    md_vdp: float,
    temp_grad: float,
    gamma_wat: float,
    angle: float,
    q_ms: float,
    d_tub: float,
    roughness: float,
) -> float:
    """Рассчитать забойное давление, Па."""
    pressure, _, _ = calc_pipe(
        p_wh,
        t_wh,
        h0,
        md_vdp,
        temp_grad,
        gamma_wat,
        angle,
        q_ms,
        d_tub,
        roughness,
    )
    return float(pressure[-1])


def main() -> None:
    """Загрузить вариант № 4, рассчитать VLP и сохранить output.json."""
    homework_dir = Path(__file__).resolve().parent.parent
    input_path = homework_dir / "input_data" / "4.json"
    output_path = Path(__file__).resolve().parent / "output.json"

    with input_path.open("r", encoding="utf-8") as file:
        input_data = json.load(file)

    # Согласно условию обе границы диапазона 1–400 м³/сут включены.
    q_liq = [1] + list(range(10, 401, 10))
    p_wf = []

    for q_m3_day in q_liq:
        p_bottom = calc_p_wf(
            p_wh=input_data["p_wh"] * ATM_TO_PA,
            t_wh=input_data["t_wh"] + 273,
            h0=0,
            md_vdp=input_data["md_vdp"],
            temp_grad=input_data["temp_grad"],
            gamma_wat=input_data["gamma_water"],
            angle=input_data["angle"],
            q_ms=q_m3_day / SECONDS_PER_DAY,
            d_tub=input_data["d_tub"],
            roughness=input_data["roughness"],
        )
        p_wf.append(p_bottom / ATM_TO_PA)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump({"q_liq": q_liq, "p_wf": p_wf}, file, indent=4)

    print(f"Сохранено: {output_path}")
    print(
        f"Точек VLP: {len(q_liq)}; "
        f"Pзаб(1)={p_wf[0]:.3f} атм; "
        f"Pзаб(400)={p_wf[-1]:.3f} атм"
    )


if __name__ == "__main__":
    main()
