import React from "react";
import "../style/loader.css";
import { Slider } from '@mui/material';

const CustomSlider = ({ value, setValue, min, max, step }) => {
    return (
        <Slider
            size="small"
            value={value}
            onChange={(e) => setValue(e)}
            aria-label="Default"
            valueLabelDisplay="auto"
            min={min}
            max={max}
            step={step}
        />
    );
};

export default CustomSlider;