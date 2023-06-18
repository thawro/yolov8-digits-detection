import labels from "../constants/labels.json"

/**
 * Render prediction boxes
 * @param {HTMLCanvasElement} canvas canvas tag reference
 * @param {Array[Object]} boxes boxes array
 */
export const renderBoxes = (imageRef, canvasRef, boxes) => {
    const canvas = canvasRef.current
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = '#FFFFFF'
    ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height)
    // ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height); // clean canvas

    const colors = new Colors();

    // font configs
    const font = `${Math.max(Math.round(Math.max(ctx.canvas.width, ctx.canvas.height) / 40), 14)}px Arial`;
    ctx.font = font;
    ctx.textBaseline = "top";
    const H = canvas.height
    const W = canvas.width

    const image = new Image();
    image.src = imageRef.current.src
    ctx.drawImage(image, 0, 0);

    boxes.forEach((box) => {
        const class_id = labels[box.label];
        const color = colors.get(box.label);
        const score = (box.conf).toFixed(2);
        const [xn, yn, wn, hn] = box.box_xywhn; // center x and y

        let x = (xn * W)
        let y = (yn * H)
        let w = wn * W
        let h = hn * H

        // xcenter,ycenter -> xmin,ymin
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
        const labelTxt = class_id + "    " + score
        const labelTextWidth = ctx.measureText(labelTxt).width;
        const TextHeight = parseInt(font, 10); // base 10
        const yText = y - (TextHeight + ctx.lineWidth);

        ctx.fillRect(
            x - 1,
            yText < 0 ? 0 : yText,
            labelTextWidth + ctx.lineWidth,
            TextHeight + ctx.lineWidth
        );

        // Draw labels
        ctx.fillStyle = "#ffffff";
        ctx.fillText(labelTxt, x - 1, yText < 0 ? 1 : yText + 1);
    });
};

export const renderInfo = (canvasRef, speed) => {
    const canvas = canvasRef.current
    const ctx = canvas.getContext("2d");

    ctx.textBaseline = "top";
    const H = canvas.height
    const W = canvas.width

    const sizeTxt = H + " x " + W

    ctx.font = `14px Arial`;
    ctx.fillStyle = "#777777"
    ctx.fillText(sizeTxt, 8, 8);

    let totalTime = 0
    for (let key in speed) {
        if (speed.hasOwnProperty(key)) {
            const time = speed[key];
            totalTime += time
        }
    }
    let fps = Math.floor(1000 / totalTime)
    let speedTxt = "Latency: " + totalTime + "ms (~" + fps + "fps)"
    ctx.fillText(speedTxt, 8, 10 + 14);

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