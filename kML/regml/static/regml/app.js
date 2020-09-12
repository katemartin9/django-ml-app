
function formatTable() {
    let elems = document.getElementsByTagName('td')
    Array.from(elems).forEach(elem => {
        debugger;
        typeof elem.innerText
    elem.innerText = Number(elem.innerText).toFixed(1);
    });
}

formatTable()