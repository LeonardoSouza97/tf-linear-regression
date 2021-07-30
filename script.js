const learn = async () => {
  console.time()
  const model = tf.sequential({
    layers: [
      tf.layers.dense({ inputShape: [1], units: 1 })
    ]
  })

  model.compile({
    loss: 'meanSquaredError',
    optimizer: 'sgd'
  })

  const valoresX = tf.tensor2d([-1, 0, 1, 2, 3, 4], [6, 1]);

  const valoresY = tf.tensor2d([-3, -1, 1, 3, 5, 7], [6, 1]);

  await model.fit(valoresX, valoresY, { epochs: 1000 })

  const resultado = model.predict(tf.tensor2d([20], [1, 1]));

  document.getElementById('resultado').innerHTML = resultado;

  console.timeEnd();
}

learn();