
function formatTable() {
    let elems = document.getElementsByTagName('td')
    Array.from(elems).forEach(elem => {
        debugger;
        typeof elem.innerText
    elem.innerText = Number(elem.innerText).toFixed(1);
    });
}

//formatTable()

function apiData(projectName, id) {
  return new Promise(function(resolve, reject) {
    d3.json(`'http://http://127.0.0.1:8000/regml/model-options/'${projectName}/${id}`, function(error, data) {
      if (error) {
      	reject(error);
      } else {
      	resolve(data);
      }
    });
  });
}

apiData('boston project', 'api')
  .then(function (data) {
    console.log(data);
  })
  .catch(function (error) {
    console.log(error);
  });