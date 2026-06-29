#!/usr/bin/env python3
"""Generate Surgical-TSplineGS Technical Solution PDF document (Chinese)."""

from fpdf import FPDF
import os

FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_BOLD_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"

class SurgicalTSplineGSPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.add_font("Noto", "", FONT_PATH, uni=True)
        self.add_font("Noto", "B", FONT_BOLD_PATH, uni=True)
        self.add_font("NotoMono", "", "/usr/share/fonts/truetype/arphic/uming.ttc", uni=True)

    def header(self):
        if self.page_no() > 1:
            self.set_font("Noto", "", 8)
            self.cell(0, 8, "Surgical-TSplineGS \u6280\u672f\u65b9\u6848\u6587\u6863", align="C")
            self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("Noto", "", 8)
        self.cell(0, 10, f"- {self.page_no()} -", align="C")

    def chapter_title(self, title, level=1):
        if level == 1:
            self.set_font("Noto", "B", 18)
            self.set_text_color(0, 51, 102)
            self.ln(6)
            self.cell(0, 12, title)
            self.ln(10)
            self.set_draw_color(0, 51, 102)
            self.set_line_width(0.6)
            self.line(10, self.get_y(), 200, self.get_y())
            self.ln(6)
        elif level == 2:
            self.set_font("Noto", "B", 14)
            self.set_text_color(0, 51, 102)
            self.ln(4)
            self.cell(0, 10, title)
            self.ln(8)
        elif level == 3:
            self.set_font("Noto", "B", 11)
            self.set_text_color(51, 51, 51)
            self.ln(2)
            self.cell(0, 8, title)
            self.ln(6)

    def body_text(self, text):
        self.set_font("Noto", "", 10)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5.5, text)
        self.ln(2)

    def bullet(self, text, indent=15):
        self.set_font("Noto", "", 10)
        self.set_text_color(30, 30, 30)
        x = self.get_x()
        self.cell(indent)
        self.cell(5, 5.5, "\u2022")
        self.multi_cell(0, 5.5, text)
        self.ln(1)

    def code_block(self, text):
        self.set_font("NotoMono", "", 8.5)
        self.set_text_color(30, 30, 30)
        self.set_fill_color(240, 240, 240)
        self.set_draw_color(200, 200, 200)
        self.ln(2)
        for line in text.split("\n"):
            self.cell(12)
            self.cell(0, 4.5, line, fill=True)
            self.ln(4.5)
        self.ln(2)

    def note_box(self, text):
        self.set_fill_color(230, 242, 255)
        self.set_draw_color(0, 102, 204)
        self.set_font("Noto", "", 9)
        self.set_text_color(0, 51, 102)
        self.ln(2)
        self.cell(12)
        self.multi_cell(0, 5, text, border=1, fill=True)
        self.ln(4)


def build_pdf():
    pdf = SurgicalTSplineGSPDF()
    pdf.set_auto_page_break(auto=True, margin=20)

    # ============================================================
    # Cover Page
    # ============================================================
    pdf.add_page()
    pdf.ln(40)
    pdf.set_font("Noto", "B", 28)
    pdf.set_text_color(0, 51, 102)
    pdf.cell(0, 14, "Surgical-TSplineGS", align="C")
    pdf.ln(16)
    pdf.set_font("Noto", "", 16)
    pdf.set_text_color(60, 60, 60)
    pdf.cell(0, 10, "Robust Motion-Adaptive Spline for Real-Time", align="C")
    pdf.ln(8)
    pdf.cell(0, 10, "Dynamic 3D Gaussians from Monocular Video", align="C")
    pdf.ln(20)
    pdf.set_draw_color(0, 51, 102)
    pdf.set_line_width(0.8)
    pdf.line(60, pdf.get_y(), 150, pdf.get_y())
    pdf.ln(14)
    pdf.set_font("Noto", "", 12)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 8, "CVPR 2025", align="C")
    pdf.ln(8)
    pdf.cell(0, 8, "BeiHang University", align="C")
    pdf.ln(14)
    pdf.set_font("Noto", "", 10)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 7, "Qi Chen", align="C")
    pdf.ln(14)
    pdf.set_font("Noto", "", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 7, "https://github.com/chenqi111/Surgical-TSplineGS", align="C")

    # ============================================================
    # TOC
    # ============================================================
    pdf.add_page()
    pdf.chapter_title("\u76ee\u5f55")
    pdf.set_font("Noto", "", 11)
    items = [
        ("1", "\u9879\u76ee\u6982\u8ff0"),
        ("2", "\u6574\u4f53\u67b6\u6784"),
        ("3", "\u6838\u5fc3\u6280\u672f"),
        ("", "  3.1 \u9759\u6001-\u52a8\u6001\u9ad8\u65af\u89e3\u8026"),
        ("", "  3.2 \u8fd0\u52a8\u81ea\u9002\u5e94\u4e09\u6b21 Hermite \u6837\u6761"),
        ("", "  3.3 \u81ea\u9002\u5e94\u63a7\u5236\u70b9\u526a\u88c1"),
        ("", "  3.4 \u76f8\u673a\u4f4d\u59ff\u4e0e\u6df1\u5ea6\u8054\u5408\u4f30\u8ba1 (CVD)"),
        ("", "  3.5 \u989c\u8272\u89e3\u7801\u5668 (RGB Decoder)"),
        ("4", "\u8bad\u7ec3\u6d41\u7a0b"),
        ("", "  4.1 Stage 1: Warm-up (\u9884\u70ed\u9636\u6bb5)"),
        ("", "  4.2 Stage 2: Fine-static (u9759\u6001\u7cbe\u8c03)"),
        ("", "  4.3 Stage 3: Fine (\u8054\u5408\u7cbe\u8c03)"),
        ("5", "\u635f\u5931\u51fd\u6570"),
        ("6", "\u6e32\u67d3\u7ba1\u7ebf"),
        ("7", "\u5b9e\u9a8c\u7ed3\u679c"),
        ("8", "\u603b\u7ed3"),
    ]
    for num, title in items:
        if num:
            pdf.set_text_color(0, 51, 102)
            pdf.set_font("Noto", "B", 11)
            pdf.cell(8, 8, num)
        else:
            pdf.set_text_color(60, 60, 60)
            pdf.set_font("Noto", "", 10)
            pdf.cell(8)
        pdf.cell(0, 8, title)
        pdf.ln(7)

    # ============================================================
    # 1. Project Overview
    # ============================================================
    pdf.add_page()
    pdf.chapter_title("1 \u9879\u76ee\u6982\u8ff0")

    pdf.body_text(
        "Surgical-TSplineGS \u662f\u4e00\u4e2a\u57fa\u4e8e 3D Gaussian Splatting (3DGS) \u7684\u52a8\u6001\u573a\u666f"
        "\u91cd\u5efa\u4e0e\u5b9e\u65f6\u6e32\u67d3\u6846\u67b6\uff0c\u7531 BeiHang University \u63d0\u51fa\u3002"
        "\u8be5\u65b9\u6cd5\u4ece\u5355\u76ee\u89c6\u9891\u4e2d\u5b66\u4e60\u52a8\u6001\u573a\u666f\u7684"
        "3D \u8868\u793a\uff0c\u5b9e\u73b0\u9ad8\u8d28\u91cf\u7684\u65b0\u89c6\u89d2\u5408\u6210\u548c\u5b9e\u65f6\u6e32\u67d3\u3002"
    )

    pdf.body_text(
        "\u6838\u5fc3\u6311\u6218\uff1a\u4ece\u5355\u76ee\u89c6\u9891\u590d\u539f\u52a8\u6001\u573a\u666f\u7684\u4e3b\u8981\u96be\u70b9\u5305\u62ec\uff1a"
        "(1) \u76f8\u673a\u8fd0\u52a8\u4e0e\u7269\u4f53\u8fd0\u52a8\u7684\u6df7\u5408\u6620\u5c04\uff1b"
        "(2) \u5355\u76ee\u89c6\u9891\u7f3a\u4e4f\u6df1\u5ea6\u4fe1\u606f\uff1b"
        "(3) \u5b9e\u65f6\u6027\u80fd\u8981\u6c42\u3002"
        "Surgical-TSplineGS \u901a\u8fc7\u4ee5\u4e0b\u521b\u65b0\u70b9\u89e3\u51b3\u8fd9\u4e9b\u95ee\u9898\uff1a"
    )

    pdf.bullet("\u9759\u6001-\u52a8\u6001\u9ad8\u65af\u89e3\u8026\u8868\u793a")
    pdf.bullet("\u8fd0\u52a8\u81ea\u9002\u5e94\u4e09\u6b21 Hermite \u6837\u6761\u63a7\u5236\u70b9\u8868\u793a\u8f68\u8ff9")
    pdf.bullet("\u76f8\u673a\u4f4d\u59ff\u4e0e\u6df1\u5ea6\u5c3a\u5ea6\u7684\u8054\u5408\u4f18\u5316 (CVD)")
    pdf.bullet("\u57fa\u4e8e gsplat \u7684\u5b9e\u65f6\u53ef\u5fae\u5206\u6e32\u67d3")

    # ============================================================
    # 2. Overall Architecture
    # ============================================================
    pdf.chapter_title("2 \u6574\u4f53\u67b6\u6784")

    pdf.body_text(
        "Surgical-TSplineGS \u7684\u6574\u4f53\u67b6\u6784\u5982\u56fe\u6240\u793a\uff0c\u4e3b\u8981\u5305\u542b\u4ee5\u4e0b\u7ec4\u4ef6\uff1a"
    )

    pdf.body_text("[Input] Monocular Video")
    pdf.body_text(
        "      \u2193"
    )
    pdf.body_text("[Preprocessing] \u2192 Co-Tracker (\u5149\u6d41\u8ffd\u8e2a) + UniDepth / Depth-Anything (\u6df1\u5ea6\u4f30\u8ba1)")
    pdf.body_text(
        "      \u2193"
    )
    pdf.body_text("[Pose Network (CVD)] \u2192 \u65f6\u5e8f MLP \u8f93\u51fa\u76f8\u673a\u4f4d\u59ff + \u5c3a\u5ea6\u6df1\u5ea6")
    pdf.body_text(
        "      \u2193"
    )
    pdf.body_text("[Gaussian Models] \u2192 Static Gaussians + Dynamic Gaussians (Hermite Spline Control Points)")
    pdf.body_text(
        "      \u2193"
    )
    pdf.body_text("[gsplat Rasterizer] \u2192 Tile-based \u53ef\u5fae\u5206\u6e32\u67d3 (RGB+ED)")
    pdf.body_text(
        "      \u2193"
    )
    pdf.body_text("[RGB Decoder] \u2192 MLP \u89e3\u7801\u5668\u8f93\u51fa\u7ec8\u6e32\u67d3\u7ed3\u679c")
    pdf.body_text(
        "      \u2193"
    )
    pdf.body_text("[Output] Novel View Synthesis + Motion Masks + Depth Maps")

    pdf.ln(4)
    pdf.note_box(
        "\u5173\u952e\u6587\u4ef6\u4f4d\u7f6e\uff1a"
        "train.py (\u8bad\u7ec3\u4e3b\u7a0b\u5e8f), scene/gaussian_model.py (\u9ad8\u65af\u6a21\u578b), "
        "scene/deformation.py (\u4f4d\u59ff\u7f51\u7edc), gaussian_renderer/__init__.py (\u6e32\u67d3\u7ba1\u7ebf)"
    )

    # ============================================================
    # 3. Core Technology
    # ============================================================
    pdf.chapter_title("3 \u6838\u5fc3\u6280\u672f")

    # 3.1 Static-Dynamic Decomposition
    pdf.chapter_title("3.1 \u9759\u6001-\u52a8\u6001\u9ad8\u65af\u89e3\u8026", level=2)
    pdf.body_text(
        "Surgical-TSplineGS \u5c06\u573a\u666f\u5206\u89e3\u4e3a\u4e24\u7ec4\u72ec\u7acb\u7684 3D Gaussians\uff1a"
    )
    pdf.bullet(
        "Static Gaussians (stat_gaussians)\uff1a\u8868\u793a\u573a\u666f\u4e2d\u4e0d\u53d8\u7684\u90e8\u5206"
        "\uff08\u5982\u80cc\u666f\u3001\u56fa\u5b9a\u7ed3\u6784\uff09\uff0c\u4f4d\u7f6e\u56fa\u5b9a\uff0c\u4e0d\u53d8\u5f62\u3002"
    )
    pdf.bullet(
        "Dynamic Gaussians (dyn_gaussians)\uff1a\u8868\u793a\u52a8\u6001\u7269\u4f53\uff0c\u6bcf\u4e2a\u70b9\u7684\u4f4d\u7f6e"
        "\u901a\u8fc7 Hermite \u6837\u6761\u63a7\u5236\u70b9\u63cf\u8ff0\u5176\u65f6\u95f4\u8f68\u8ff9\u3002"
    )
    pdf.body_text(
        "\u6e32\u67d3\u65f6\u5c06\u4e24\u7ec4\u9ad8\u65af\u62fc\u63a5\u540e\u4e00\u8d77\u8fdb\u884c\u6e32\u67d3\uff0c\u5e76\u901a\u8fc7\u4e00\u4e2a alpha \u901a\u9053\u56fe"
        "(d_alpha) \u6765\u533a\u5206\u52a8\u9759\u533a\u57df\u3002"
        "\u8fd9\u79cd\u89e3\u8026\u8868\u793a\u51cf\u5c11\u4e86\u52a8\u6001\u8868\u793a\u7684\u590d\u6742\u5ea6\uff0c\u56e0\u4e3a\u9759\u6001\u90e8\u5206\u4e0d\u9700\u8981\u65f6\u95f4\u7ef4\u5ea6\u7684\u53d8\u5f62\u3002"
    )
    pdf.code_block(
        "# gaussian_renderer/__init__.py - Combining static and dynamic Gaussians\n"
        "means3D_final = torch.cat((smeans3D_final, means3D_final), 0)\n"
        "scales_final  = torch.cat((sscales_final,  scales_final),  0)\n"
        "rotations_final = torch.cat((srotations_final, rotations_final), 0)\n"
        "opacity_final = torch.cat((sopacity_final, opacity_final), 0)\n"
        "colors_precomp_final = torch.cat((stat_colors_precomp, colors_precomp), 0)"
    )

    # 3.2 Cubic Hermite Spline
    pdf.chapter_title("3.2 \u8fd0\u52a8\u81ea\u9002\u5e94\u4e09\u6b21 Hermite \u6837\u6761", level=2)
    pdf.body_text(
        "\u4e0e\u4f20\u7edf\u65b9\u6cd5\u4f7f\u7528 MLP \u6216\u795e\u7ecf\u7f51\u7edc\u8868\u793a\u8fd0\u52a8\u573a\u4e0d\u540c\uff0c"
        "Surgical-TSplineGS \u4e3a\u6bcf\u4e2a\u52a8\u6001\u9ad8\u65af\u70b9\u5206\u914d\u4e00\u7ec4\u53ef\u5b66\u4e60\u7684\u63a7\u5236\u70b9 (control_xyz)\uff0c"
        "\u901a\u8fc7\u4e09\u6b21\u5bc6\u96c6\u63d2\u503c (Cubic Hermite Spline) \u5728\u4efb\u610f\u65f6\u95f4\u6b65\u6c42\u89e3\u4f4d\u79fb\u504f\u5dee\u3002"
    )
    pdf.body_text(
        "\u63a7\u5236\u70b9\u6570\u91cf\u4e3a ctrl_num\uff08\u9ed8\u8ba4 15\uff09\uff0c\u901a\u8fc7\u5c06\u8f68\u8ff9\u7ebf\u6027\u62df\u5408\u5230\u4e00\u7ec4\u7a00\u758f\u7684\u63a7\u5236\u70b9\u4e0a\u3002"
        "\u63d2\u503c\u516c\u5f0f\u5982\u4e0b\uff1a"
    )
    pdf.body_text(
        "  interpolation = h00 * p1 + h10 * m0 + h01 * p2 + h11 * m1\n"
        "\u5176\u4e2d h00, h10, h01, h11 \u4e3a Hermite \u57fa\u51fd\u6570\uff0c"
        "p0/p1/p2/p3 \u4e3a\u76f8\u90bb\u63a7\u5236\u70b9\uff0cm0/m1 \u4e3a\u5355\u4fa7\u5bfc\u6570\u3002"
    )
    pdf.body_text(
        "\u6837\u6761\u63d2\u503c\u5728\u4e24\u4e2a\u5730\u65b9\u5b9e\u73b0\uff1a"
    )
    pdf.bullet("gaussian_renderer/__init__.py: interpolate_cubic_hermite() — \u8bad\u7ec3\u65f6\u4f7f\u7528")
    pdf.bullet("scene/gaussian_model.py: interpolate_cubic_hermite() — \u63a7\u5236\u70b9\u7ba1\u7406\u548c\u526a\u88c1\u65f6\u4f7f\u7528")
    pdf.body_text(
        "\u6bcf\u4e2a\u52a8\u6001\u9ad8\u65af\u70b9\u7684\u6700\u7ec8\u4f4d\u7f6e\u4e3a\uff1a"
        "means3D = deform_means3D * deform_spatial_scale\uff0c\u5373\u57fa\u7840\u4f4d\u7f6e\u4e0a\u53e0\u52a0\u6837\u6761\u63d2\u503c\u7684\u504f\u5dee\u3002"
    )

    # 3.3 Adaptive Control Point Pruning
    pdf.chapter_title("3.3 \u81ea\u9002\u5e94\u63a7\u5236\u70b9\u526a\u88c1", level=2)
    pdf.body_text(
        "Surgical-TSplineGS \u7684\u91cd\u8981\u521b\u65b0\u70b9\u662f\u8fd0\u52a8\u81ea\u9002\u5e94\u7684\u63a7\u5236\u70b9\u6570\u91cf\u7ba1\u7406\u3002"
        "\u5728\u8bad\u7ec3\u8fc7\u7a0b\u4e2d\uff0c\u7cfb\u7edf\u4f1a\u901a\u8fc7 onedown_control_pts() \u9010\u6b65\u51cf\u5c11\u63a7\u5236\u70b9\u6570\u91cf\uff1a"
    )
    pdf.bullet("\u6bcf\u9694\u4e00\u5b9a\u8bad\u7ec3\u8fed\u4ee3\u6b65\u6570\u6267\u884c\u4e00\u6b21\u63a7\u5236\u70b9\u51cf\u5c11\u64cd\u4f5c")
    pdf.bullet("\u7528 Hermite \u53cd\u6f14\u7b97\u6cd5\u5c06\u63a7\u5236\u70b9\u6570\u4ece N \u51cf\u5c11\u5230 N-1")
    pdf.bullet("\u901a\u8fc7\u8ba1\u7b97 2D \u6295\u5f71\u8bef\u5dee (compute_prune_error) \u6765\u8bc4\u4f30\u63a7\u5236\u70b9\u51cf\u5c11\u5bf9\u8d28\u91cf\u7684\u5f71\u54cd")
    pdf.bullet("\u4ec5\u5bf9\u8bef\u5dee\u4f4e\u4e8e\u9608\u503c\u7684\u9ad8\u65af\u70b9\u6267\u884c\u526a\u88c1\uff0c\u590d\u6742\u8fd0\u52a8\u4fdd\u7559\u66f4\u591a\u63a7\u5236\u70b9")
    pdf.body_text(
        "\u8fd9\u6837\uff0c\u521a\u6027\u8fd0\u52a8\uff08\u5982\u7269\u4f53\u6574\u4f53\u79fb\u52a8\uff09\u53ea\u9700\u8981\u5f88\u5c11\u63a7\u5236\u70b9\uff0c\u800c\u590d\u6742\u975e\u521a\u6027\u8fd0\u52a8"
        "\uff08\u5982\u8863\u7269\u6446\u52a8\uff09\u4fdd\u7559\u66f4\u591a\u63a7\u5236\u70b9\uff0c\u5b9e\u73b0\u4e86\u8868\u793a\u6548\u7387\u4e0e\u8868\u73b0\u529b\u7684\u5e73\u8861\u3002"
    )

    # 3.4 Pose Network (CVD)
    pdf.chapter_title("3.4 \u76f8\u673a\u4f4d\u59ff\u4e0e\u6df1\u5ea6\u8054\u5408\u4f30\u8ba1 (CVD)", level=2)
    pdf.body_text(
        "Pose Network (pose_network) \u662f\u4e00\u4e2a\u7b80\u5355\u7684 MLP\uff0c\u4f4d\u4e8e scene/deformation.py\uff0c"
        "\u5b83\u4ee5\u65f6\u95f4\u7f16\u7801 (positional encoding of time) \u4e3a\u8f93\u5165\uff0c"
        "\u8054\u5408\u4f30\u8ba1\u76f8\u673a\u5916\u53c2\uff08Euler \u89d2 + \u5e73\u79fb\uff09\u548c\u5c3a\u5ea6\u6df1\u5ea6\uff08CVD: Canonical View Depth\uff09\u3002"
    )
    pdf.body_text("\u7f51\u7edc\u7ed3\u6784\uff1a")
    pdf.bullet("Timenet: \u4e24\u5c42 MLP (128 -> 128) \u5904\u7406\u65f6\u95f4\u7f16\u7801\u7279\u5f81")
    pdf.bullet("Pose output: \u7ebf\u6027\u5c42\u8f93\u51fa 6 \u4e2a\u53c2\u6570 (3 \u4e2a Euler \u89d2 + 3 \u4e2a\u5e73\u79fb)")
    pdf.bullet("Depth output: \u7ebf\u6027\u5c42\u8f93\u51fa\u5c3a\u5ea6\u56e0\u5b50 (instance_scale)")
    pdf.bullet("Focal bias: \u53ef\u5b66\u4e60\u7684\u7126\u8ddd\u53c2\u6570 (focal_bias)")
    pdf.body_text(
        "CVD = depth * instance_scale / cannonical_scale\uff0c\u5373\u901a\u8fc7\u5b66\u4e60\u6bcf\u5e27\u7684\u5c3a\u5ea6\u56e0\u5b50\u6765\u89e3\u51b3\u5355\u76ee\u89c6\u9891\u7684\u6df1\u5ea6\u6a21\u7cca\u6027\u3002"
        "\u7f51\u7edc\u53ef\u4ee5\u5c06\u9884\u6d4b\u7684\u6df1\u5ea6\u56fe\u7ffb\u8f6c\u4e3a 3D \u70b9\u4e91\uff08points_from_DRTK\uff09\uff0c"
        "\u7528\u4e8e\u6784\u5efa\u521d\u59cb\u9ad8\u65af\u70b9\u4e91\u548c\u8ba1\u7b97\u51e0\u4f55\u4e00\u81f4\u6027\u635f\u5931\u3002"
    )

    # 3.5 RGB Decoder
    pdf.chapter_title("3.5 \u989c\u8272\u89e3\u7801\u5668 (RGB Decoder)", level=2)
    pdf.body_text(
        "gsplat \u6e32\u67d3\u5668\u8f93\u51fa\u7684\u989c\u8272\u662f\u4e34\u65f6\u7684 intermediate \u8868\u793a\uff0c"
        "\u9700\u8981\u7ecf\u8fc7\u4e00\u4e2a\u8f7b\u91cf\u7ea7 MLP \u89e3\u7801\u5668 (getcolormodel) \u8f6c\u6362\u4e3a\u6700\u7ec8\u7684 RGB \u989c\u8272\u3002"
        "\u89e3\u7801\u5668\u8fd8\u63a5\u53d7\u76f8\u673a\u5149\u7ebf\u65b9\u5411 (cam_ray) \u4f5c\u4e3a\u8f93\u5165\uff0c"
        "\u4f7f\u5f97\u989c\u8272\u8868\u793a\u5177\u6709\u89c6\u89d2\u4f9d\u8d56\u6027\u3002"
    )

    # ============================================================
    # 4. Training Pipeline
    # ============================================================
    pdf.add_page()
    pdf.chapter_title("4 \u8bad\u7ec3\u6d41\u7a0b")
    pdf.body_text(
        "Surgical-TSplineGS \u91c7\u7528\u4e09\u9636\u6bb5\u8bad\u7ec3\u7b56\u7565\uff0c\u6bcf\u9636\u6bb5\u4f18\u5316\u4e0d\u540c\u7684\u53c2\u6570\u96c6\uff1a"
    )

    # 4.1 Warm-up
    pdf.chapter_title("4.1 Stage 1: Warm-up (\u9884\u70ed\u9636\u6bb5)", level=2)
    pdf.body_text(
        "\u76ee\u6807\uff1a\u4f18\u5316\u76f8\u673a\u4f4d\u59ff\u3001\u7126\u8ddd\u3001\u6df1\u5ea6\u5c3a\u5ea6\u548c\u8ffd\u8e2a\u70b9\u8f68\u8ff9"
    )
    pdf.bullet("\u8bad\u7ec3 Iterations\uff1a\u7ea6 3000-5000 \u6b65")
    pdf.bullet("\u6d3b\u8dc3\u53c2\u6570\uff1apose_network (MLP \u548c scale) + focal_bias")
    pdf.bullet("\u9ad8\u65af\u4f53\u672a\u521d\u59cb\u5316\uff0c\u4ec5\u8fdb\u884c\u4f4d\u59ff\u4f30\u8ba1")
    pdf.bullet(
        "\u635f\u5931\u51fd\u6570\uff1a\u57fa\u4e8e\u76f8\u5bf9\u4f4d\u59ff\u7684\u5149\u5b66\u4e00\u81f4\u6027 (photometric loss) +"
        "\u8ffd\u8e2a\u70b9\u635f\u5931 (track loss, \u6765\u81ea Co-Tracker \u7684 pixel \u8f68\u8ff9\u76d1\u7763)"
    )
    pdf.body_text(
        "\u8be5\u9636\u6bb5\u7684\u6838\u5fc3\u662f\u5229\u7528\u76f8\u90bb\u5e27\u7684\u76f8\u5bf9\u4f4d\u59ff\u8fdb\u884c\u53cd\u5411 warp\uff0c"
        "\u901a\u8fc7\u5149\u5b66\u4e00\u81f4\u6027 + \u8ffd\u8e2a\u70b9\u76d1\u7763\u6765\u5b66\u4e60\u76f8\u673a\u8fd0\u52a8\u548c\u6df1\u5ea6\u5c3a\u5ea6\u3002"
        "\u5728\u6b64\u9636\u6bb5\u7ed3\u675f\u65f6\uff0c\u4ece CVD \u6df1\u5ea6\u548c\u4f4d\u59ff\u89e3\u6790\u521d\u59cb\u52a8\u6001/\u9759\u6001\u9ad8\u65af\u70b9\u4e91\u3002"
    )

    # 4.2 Fine-static
    pdf.chapter_title("4.2 Stage 2: Fine-static (\u9759\u6001\u7cbe\u8c03)", level=2)
    pdf.body_text(
        "\u76ee\u6807\uff1a\u4ec5\u4f18\u5316\u9759\u6001\u9ad8\u65af\u53c2\u6570\uff0c\u52a8\u6001\u90e8\u5206\u51bb\u7ed3"
    )
    pdf.bullet("\u8bad\u7ec3 Iterations\uff1a\u7ea6 3000-5000 \u6b65")
    pdf.bullet("\u6d3b\u8dc3\u53c2\u6570\uff1astatic Gaussians \u7684\u6240\u6709\u5c5e\u6027\uff08xyz, scale, rotation, opacity, SH\uff09")
    pdf.bullet("\u4f7f\u7528 motion mask \u9690\u85cf\u52a8\u6001\u533a\u57df\u7684\u635f\u5931")
    pdf.bullet("\u8fd0\u884c\u5bc6\u96c6\u5316 (densification) \u548c\u526a\u88c1 (pruning) \u64cd\u4f5c")

    # 4.3 Fine
    pdf.chapter_title("4.3 Stage 3: Fine (\u8054\u5408\u7cbe\u8c03)", level=2)
    pdf.body_text(
        "\u76ee\u6807\uff1a\u8054\u5408\u4f18\u5316\u6240\u6709\u53c2\u6570"
    )
    pdf.bullet("\u8bad\u7ec3 Iterations\uff1a\u7ea6 20000-30000 \u6b65")
    pdf.bullet("\u6d3b\u8dc3\u53c2\u6570\uff1a\u52a8\u6001/\u9759\u6001 Gaussians + pose_network + RGB decoder")
    pdf.bullet("\u8fd0\u884c\u52a8\u6001\u9ad8\u65af\u7684\u5bc6\u96c6\u5316\u548c\u63a7\u5236\u70b9\u526a\u88c1 (onedown_control_pts)")
    pdf.bullet("\u4f7f\u7528\u5b8c\u6574\u7684\u635f\u5931\u51fd\u6570\u96c6\uff08\u89c1\u7b2c 5 \u8282\uff09")
    pdf.note_box(
        "Stage 3 \u7684\u8bad\u7ec3\u5faa\u73af\u4e2d\uff0c\u6bcf\u6b65\u91c7\u6837 batch_size \u4e2a\u89c6\u56fe\uff0c"
        "\u6bcf\u4e2a\u89c6\u56fe\u914d\u4e00\u4e2a\u524d\u5411\u548c\u540e\u5411\u53c2\u8003\u5e27\uff0c"
        "\u7528\u4e8e\u8ba1\u7b97\u5149\u5b66\u548c\u51e0\u4f55\u4e00\u81f4\u6027\u635f\u5931\u3002"
    )

    # ============================================================
    # 5. Loss Functions
    # ============================================================
    pdf.add_page()
    pdf.chapter_title("5 \u635f\u5931\u51fd\u6570")
    pdf.body_text(
        "Surgical-TSplineGS \u7684\u635f\u5931\u51fd\u6570\u5305\u542b\u591a\u4e2a\u7ec4\u4ef6\uff0c\u7528\u4e8e\u7efc\u5408\u4f18\u5316\u573a\u666f\u8868\u793a\u3001"
        "\u76f8\u673a\u4f4d\u59ff\u548c\u52a8\u6001\u8f68\u8ff9\uff1a"
    )
    pdf.ln(2)

    pdf.chapter_title("\u5149\u5b66\u91cd\u5efa\u635f\u5931 (Photometric Loss)", level=3)
    pdf.body_text("L_photo = L1 + lambda_dssim * (1 - SSIM)")
    pdf.body_text(
        "\u4e3b\u8981\u7684\u56fe\u50cf\u8d28\u91cf\u635f\u5931\uff0c\u7528\u4e8e\u76d1\u7763\u9ad8\u65af\u6e32\u67d3\u7ed3\u679c\u4e0e GT \u56fe\u50cf\u7684\u4e00\u81f4\u6027\u3002"
        "\u5728 warm-up \u9636\u6bb5\u4f1a\u52a0\u5165 motion mask \u9690\u85cf\u52a8\u6001\u533a\u57df\u3002"
    )

    pdf.chapter_title("\u4f4d\u59ff\u4e00\u81f4\u6027\u635f\u5931 (Pose Consistency Loss)", level=3)
    pdf.body_text(
        "\u8be5\u635f\u5931\u57fa\u4e8e\u76f8\u90bb\u5e27\u7684\u53cd\u5411 warp\uff0c\u5305\u542b\u4e09\u4e2a\u90e8\u5206\uff1a"
    )
    pdf.bullet(
        "Color loss: warp \u540e\u7684\u56fe\u50cf\u4e0e\u76ee\u6807\u56fe\u50cf\u7684\u5149\u5b66\u4e00\u81f4\u6027"
        "\uff0c\u7528\u4e8e\u4f18\u5316\u76f8\u673a\u4f4d\u59ff"
    )
    pdf.bullet(
        "Geometry loss: CVD \u6df1\u5ea6\u91cd\u6295\u5f71\u7684 3D \u4e00\u81f4\u6027\uff0c"
        "\u786e\u4fdd\u6df1\u5ea6\u4f30\u8ba1\u7684\u51e0\u4f55\u5408\u7406\u6027"
    )
    pdf.bullet(
        "\u81ea\u9002\u5e94\u906e\u7f69\uff1ags_mask, color_mask, geo_mask \u548c\u53ef\u89c1\u6027 (occ_map) \u7528\u4e8e"
        "\u8fc7\u6ee4\u51fa\u906e\u6321\u3001\u5927\u8bef\u5dee\u548c\u52a8\u6001\u533a\u57df"
    )

    pdf.chapter_title("\u6df1\u5ea6\u635f\u5931 (Depth Loss)", level=3)
    pdf.body_text("L_depth = w_depth * L1(depth_tensor, gt_depth_tensor)")
    pdf.body_text(
        "\u76d1\u7763\u6e32\u67d3\u6df1\u5ea6\u4e0e\u9884\u6d4b\u6df1\u5ea6 (CVD) \u7684\u4e00\u81f4\u6027\uff0c\u4ec5\u5728 fine \u9636\u6bb5\u4f7f\u7528\u3002"
    )

    pdf.chapter_title("\u6cd5\u7ebf\u635f\u5931 (Normal Loss)", level=3)
    pdf.body_text("L_normal = w_normal * L2(pred_normal, gt_normal, mask=motion_mask)")
    pdf.body_text(
        "\u4ece\u6df1\u5ea6\u56fe\u6c42\u5bfc\u8ba1\u7b97\u6cd5\u7ebf\u5411\u91cf\uff0c\u76d1\u7763\u6cd5\u7ebf\u4e00\u81f4\u6027\uff0c\u63d0\u5347\u51e0\u4f55\u8d28\u91cf\u3002"
    )

    pdf.chapter_title("Mask \u635f\u5931 (Mask Loss)", level=3)
    pdf.body_text("L_mask = w_mask * DiceLoss(d_alpha, motion_mask)")
    pdf.body_text(
        "\u7528\u4e8e\u76d1\u7763\u52a8\u6001\u9ad8\u65af\u7684 alpha \u63a9\u56fe\u4e0e\u9884\u8ba1\u7684\u8fd0\u52a8\u63a9\u56fe\u7684\u4e00\u81f4\u6027\u3002"
    )

    pdf.chapter_title("\u8ffd\u8e2a\u70b9\u635f\u5931 (Track Loss)", level=3)
    pdf.body_text(
        "L_track = w_track * (grid_sample(p_grid, current_track) - prev_track)^2 + (grid_sample(n_grid, current_track) - next_track)^2"
    )
    pdf.body_text(
        "\u4ec5\u5728 warm-up \u9636\u6bb5\u4f7f\u7528\uff0c\u5229\u7528 Co-Tracker \u751f\u6210\u7684 pixel \u8f68\u8ff9\u6765\u76d1\u7763\u4f4d\u59ff\u4f30\u8ba1\u3002"
    )

    # ============================================================
    # 6. Rendering Pipeline
    # ============================================================
    pdf.add_page()
    pdf.chapter_title("6 \u6e32\u67d3\u7ba1\u7ebf")
    pdf.body_text(
        "Surgical-TSplineGS \u7684\u6e32\u67d3\u7ba1\u7ebf\u57fa\u4e8e gsplat \u5e93\u7684\u7816\u5757\u5316\u53ef\u5fae\u5206\u6e32\u67d3\u5668 (tile-based rasterizer)\uff0c"
        "\u5177\u4f53\u6d41\u7a0b\u5982\u4e0b\uff1a"
    )
    pdf.ln(2)

    steps = [
        "\u83b7\u53d6\u52a8\u6001\u9ad8\u65af\u7684\u63a7\u5236\u70b9\uff0c\u901a\u8fc7 Hermite \u6837\u6761\u63d2\u503c\u8ba1\u7b97\u5f53\u524d\u65f6\u523b\u7684\u4f4d\u7f6e\u504f\u5dee",
        "\u5e94\u7528\u6fc0\u6d3b\u51fd\u6570\uff1ascaling = exp(_scaling), opacity = sigmoid(_opacity), rotation = normalize(_rotation)",
        "\u52a8\u6001\u9ad8\u65af\u4f4d\u7f6e\u66f4\u65b0\uff1ameans3D = deform_means3D * deform_spatial_scale",
        "\u6269\u5c55 SH \u7279\u5f81\uff1afeatures = concat(_features_dc, delta_t * _features_t)",
        "\u62fc\u63a5\u9759\u6001\u548c\u52a8\u6001\u9ad8\u65af\u4e3a\u4e00\u4e2a\u6574\u4f53\u5411\u91cf\uff0c\u8f93\u5165 gsplat rasterizer",
        "gsplat \u8f93\u51fa\u4e34\u65f6\u989c\u8272 + \u6df1\u5ea6 + 2D \u4fe1\u606f",
        "\u4e34\u65f6\u989c\u8272\u7ecf\u8fc7 RGB decoder (MLP + cam_ray) \u8f6c\u6362\u4e3a\u6700\u7ec8\u989c\u8272",
        "\u5e76\u884c\u6e32\u67d3 d_alpha (\u52a8\u6001\u533a\u57df\u6307\u793a\u5668)",
    ]
    for i, s in enumerate(steps, 1):
        pdf.bullet(f"{i}. {s}")

    pdf.ln(2)
    pdf.body_text(
        "\u6e32\u67d3\u8f93\u51fa\u5305\u542b\uff1aRGB \u56fe\u50cf\u3001\u6df1\u5ea6\u56fe\u3001\u52a8\u6001 alpha \u63a9\u56fe\u3001"
        "\u9759\u6001\u6e32\u67d3\u7ed3\u679c\u3001\u63a7\u5236\u70b9\u6570\u91cf\u70ed\u529b\u56fe\u3001"
        "\u8fd0\u52a8\u7c7b\u578b\u5206\u7c7b\u56fe\u7b49\u3002"
    )

    pdf.chapter_title("\u8fd0\u52a8\u7c7b\u578b\u5206\u7c7b", level=3)
    pdf.body_text(
        "\u6e32\u67d3\u7ba1\u7ebf\u8fd8\u5305\u542b\u4e00\u4e2a\u81ea\u52a8\u7684\u8fd0\u52a8\u5206\u7c7b\u529f\u80fd\uff1a"
    )
    pdf.bullet("\u6839\u636e\u63a7\u5236\u70b9\u8ba1\u7b97\u8fd0\u52a8\u5f20\u91cf\u3001\u521a\u6027\u5ea6\u3001\u901f\u5ea6\u4e00\u81f4\u6027\u7b49\u6307\u6807")
    pdf.bullet("\u5c06\u6bcf\u4e2a\u52a8\u6001\u9ad8\u65af\u5206\u7c7b\u4e3a\uff1astatic=0, tissue=0.5, instrument=1.0")
    pdf.bullet("\u6e32\u67d3\u51fa\u8fd0\u52a8\u7c7b\u578b\u56fe (motion_type_label)")

    # ============================================================
    # 7. Experimental Results
    # ============================================================
    pdf.chapter_title("7 \u5b9e\u9a8c\u7ed3\u679c")
    pdf.body_text(
        "\u4ee5\u4e0b\u662f Surgical-TSplineGS \u5728\u516c\u5f00\u6570\u636e\u96c6\u4e0a\u7684\u4e3b\u8981\u5b9e\u9a8c\u7ed3\u679c\uff1a"
    )
    pdf.ln(2)

    pdf.chapter_title("Nvidia RoDynRF Dataset", level=3)
    pdf.body_text(
        "Surgical-TSplineGS \u5728 12 \u4e2a Nvidia \u573a\u666f\u4e0a\u8fdb\u884c\u4e86\u6d4b\u8bd5\uff0c\u4e0e RoDynRF\u3001K-Planes\u3001"
        "HexPlane\u3001T-NeRF\u30014DGS \u7b49\u65b9\u6cd5\u8fdb\u884c\u4e86\u5bf9\u6bd4\u3002"
    )
    pdf.body_text("\u4e3b\u8981\u6307\u6807\uff1a")
    pdf.bullet("PSNR: \u5c45\u6240\u6709\u65b9\u6cd5\u4e4b\u9996\uff0c\u5927\u5e45\u8d85\u8d8a 4DGS \u548c RoDynRF")
    pdf.bullet("SSIM \u548c LPIPS: \u540c\u6837\u5904\u4e8e\u9886\u5148\u6c34\u5e73")
    pdf.bullet("\u6e32\u67d3\u901f\u5ea6: \u5b9e\u65f6\u7ea7 (RTX 3090 \u4e0a >30 FPS)")
    pdf.bullet("\u8bad\u7ec3\u65f6\u95f4: \u6bcf\u4e2a\u573a\u666f\u7ea6 30 \u5206\u949f")

    pdf.chapter_title("\u6d88\u878d\u5b9e\u9a8c (Ablation Study)", level=3)
    pdf.body_text("\u6838\u5fc3\u6d88\u878d\u5b9e\u9a8c\u8868\u660e\uff1a")
    pdf.bullet("Hermite spline \u4f18\u4e8e\u7ebf\u6027\u63d2\u503c\u548c\u591a\u9879\u5f0f\u62df\u5408")
    pdf.bullet("\u81ea\u9002\u5e94\u63a7\u5236\u70b9\u526a\u88c1\u63d0\u5347\u4e86\u8868\u793a\u6548\u7387\uff0c\u540c\u65f6\u4fdd\u6301\u8d28\u91cf")
    pdf.bullet("Stat-dyn \u89e3\u8026\u663e\u8457\u63d0\u5347\u4e86\u9759\u6001\u533a\u57df\u7684\u6e32\u67d3\u8d28\u91cf")
    pdf.bullet("CVD + pose consistency loss \u662f\u7a33\u5b9a\u4f4d\u59ff\u4f30\u8ba1\u7684\u5173\u952e")

    # ============================================================
    # 8. Summary
    # ============================================================
    pdf.add_page()
    pdf.chapter_title("8 \u603b\u7ed3")
    pdf.body_text(
        "Surgical-TSplineGS \u662f\u4e00\u4e2a\u5c06\u4f20\u7edf\u8ba1\u7b97\u56fe\u5f62\u5b66\u7684\u6837\u6761\u63d2\u503c\u4e0e\u73b0\u4ee3\u795e\u7ecf\u8868\u793a"
        "\u7ed3\u5408\u7684\u521b\u65b0\u65b9\u6cd5\uff0c\u4e3b\u8981\u8d21\u732e\u5305\u62ec\uff1a"
    )
    pdf.ln(2)

    contributions = [
        "\u63d0\u51fa\u4e86\u8fd0\u52a8\u81ea\u9002\u5e94\u7684\u4e09\u6b21\u5bc6\u96c6\u6837\u6761\u8868\u793a\u6765\u5efa\u6a21\u52a8\u6001\u9ad8\u65af\u7684\u65f6\u95f4\u8f68\u8ff9\uff0c"
        "\u6bd4 MLP \u57fa\u7840\u7684\u65b9\u6cd5\u66f4\u7b80\u5355\u3001\u66f4\u6548\u7387\uff0c\u5e76\u4e14\u66f4\u5bb9\u6613\u7406\u89e3\u548c\u8c03\u8bd5",
        "\u5b9e\u73b0\u4e86\u63a7\u5236\u70b9\u6570\u91cf\u7684\u81ea\u9002\u5e94\u7ba1\u7406\uff0c\u786e\u4fdd\u8868\u793a\u590d\u6742\u5ea6\u4e0e\u8fd0\u52a8\u590d\u6742\u5ea6\u7684\u5339\u914d",
        "\u8bbe\u8ba1\u4e86\u4e00\u5957\u5b8c\u6574\u7684\u8054\u5408\u4f18\u5316\u76f8\u673a\u4f4d\u59ff\u3001\u6df1\u5ea6\u5c3a\u5ea6\u548c 3D \u573a\u666f\u8868\u793a\u7684\u6846\u67b6",
        "\u5728\u591a\u4e2a\u516c\u5f00\u6570\u636e\u96c6\u4e0a\u8fbe\u5230\u4e86\u6700\u5148\u8fdb\u7684\u65b0\u89c6\u89d2\u5408\u6210\u8d28\u91cf\uff0c\u5e76\u652f\u6301\u5b9e\u65f6\u6e32\u67d3",
    ]
    for c in contributions:
        pdf.bullet(c)

    pdf.ln(4)
    pdf.note_box(
        "\u76f8\u5173\u94fe\u63a5\uff1a\n"
        "\u2022 \u8bba\u6587: https://arxiv.org/abs/2412.09982\n"
        "\u2022 \u9879\u76ee\u9875: https://github.com/chenqi111/Surgical-TSplineGS\n"
        "\u2022 GitHub: https://github.com/chenqi111/Surgical-TSplineGS\n"
        "\u2022 \u6a21\u578b\u67b6\u6784\u56fe: assets/architecture.png"
    )

    # ============================================================
    # Save to file
    # ============================================================
    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "Surgical_TSplineGS_Technical_Solution.pdf",
    )
    pdf.output(out_path)
    print(f"PDF generated: {out_path}")


if __name__ == "__main__":
    build_pdf()
