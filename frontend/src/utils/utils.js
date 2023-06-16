
var getSize = function (el) {
    var el_style = window.getComputedStyle(el),
        el_display = el_style.display,
        el_position = el_style.position,
        el_visibility = el_style.visibility,
        el_max_height = el_style.maxHeight.replace('px', '').replace('%', ''),
        el_max_width = el_style.maxWidth.replace('px', '').replace('%', ''),
        wantedHeight = 0,
        wantedWidth = 0;

    // if its not hidden we just return normal height
    if (el_display !== 'none' && el_max_height !== '0' && el_max_width !== '0') {
        return { height: el.offsetHeight, width: el.offsetWidth };
    }

    // the element is hidden so:
    // making the el block so we can meassure its height but still be hidden
    el.style.position = 'absolute';
    el.style.visibility = 'hidden';
    el.style.display = 'block';

    wantedHeight = el.offsetHeight;
    wantedWidth = el.offsetWidth;


    // reverting to the original values
    el.style.display = el_display;
    el.style.position = el_position;
    el.style.visibility = el_visibility;

    return { height: wantedHeight, width: wantedWidth }
}

export { getSize }