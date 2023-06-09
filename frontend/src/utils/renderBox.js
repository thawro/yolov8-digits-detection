import labels from "./labels.json";

/**
 * Render prediction boxes
 * @param {HTMLCanvasElement} canvas canvas tag reference
 * @param {Array[Object]} boxes boxes array
 */
export const renderBoxes = (canvas, boxes) => {
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height); // clean canvas

    const colors = new Colors();

    // font configs
    const font = `${Math.max(Math.round(Math.max(ctx.canvas.width, ctx.canvas.height) / 40), 12)}px Arial`;
    ctx.font = font;
    ctx.textBaseline = "top";
    const H = canvas.height
    const W = canvas.width

    boxes.forEach((box) => {
        const class_id = labels[box.label];
        const color = colors.get(box.label);
        const score = (box.probability).toFixed(1);
        const [xn, yn, wn, hn] = box.boundingNormalized; // center x and y

        let x = (xn * W)
        let y = (yn * H)
        let w = wn * W
        let h = hn * H

        x = x - w / 2
        y = y - h / 2


        ctx.fillStyle = Colors.hexToRgba(color, 0.2);
        ctx.fillRect(x, y, w, h);
        // draw border box
        ctx.strokeStyle = color;
        ctx.lineWidth = Math.max(Math.min(ctx.canvas.width, ctx.canvas.height) / 200, 2.5);
        ctx.strokeRect(x, y, w, h);

        // draw the label background.
        ctx.fillStyle = color;
        const txt = class_id + "  " + score
        const textWidth = ctx.measureText(txt).width;
        const textHeight = parseInt(font, 10); // base 10
        const yText = y - (textHeight + ctx.lineWidth);
        ctx.fillRect(
            x - 1,
            yText < 0 ? 0 : yText,
            textWidth + ctx.lineWidth,
            textHeight + ctx.lineWidth
        );

        // Draw labels
        ctx.fillStyle = "#ffffff";
        ctx.fillText(txt, x - 1, yText < 0 ? 1 : yText + 1);
    });
};

class Colors {
    // ultralytics color palette https://ultralytics.com/
    constructor() {
        this.palette = [
            "#FF3838",
            "#FF9D97",
            "#FF701F",
            "#FFB21D",
            "#CFD231",
            "#48F90A",
            "#92CC17",
            "#3DDB86",
            "#1A9334",
            "#00D4BB",
            "#2C99A8",
            "#00C2FF",
            "#344593",
            "#6473FF",
            "#0018EC",
            "#8438FF",
            "#520085",
            "#CB38FF",
            "#FF95C8",
            "#FF37C7",
        ];
        this.n = this.palette.length;
    }

    get = (i) => this.palette[Math.floor(i) % this.n];

    static hexToRgba = (hex, alpha) => {
        var result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
        return result
            ? `rgba(${[parseInt(result[1], 16), parseInt(result[2], 16), parseInt(result[3], 16)].join(
                ", "
            )}, ${alpha})`
            : null;
    };
}