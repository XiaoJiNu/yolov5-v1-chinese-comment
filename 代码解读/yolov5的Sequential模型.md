Sequential(
  (0): Focus(
    (conv): Conv(
      (conv): Conv2d(12, 80, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(80, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
      (act): LeakyReLU(negative_slope=0.1, inplace=True)
    )
  )
  (1): Conv(
    (conv): Conv2d(80, 160, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    (bn): BatchNorm2d(160, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
    (act): LeakyReLU(negative_slope=0.1, inplace=True)
  )
  (2): BottleneckCSP(
    (cv1): Conv(
      (conv): Conv2d(160, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(80, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
      (act): LeakyReLU(negative_slope=0.1, inplace=True)
    )
    (cv2): Conv2d(160, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (cv3): Conv2d(80, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (cv4): Conv(
      (conv): Conv2d(160, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(160, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
      (act): LeakyReLU(negative_slope=0.1, inplace=True)
    )
    (bn): BatchNorm2d(160, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
    (act): LeakyReLU(negative_slope=0.1, inplace=True)
    (m): Sequential(
      (0): Bottleneck(
        (cv1): Conv(
          (conv): Conv2d(80, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(80, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (cv2): Conv(
          (conv): Conv2d(80, 80, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(80, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
      )
      (1): Bottleneck(
        (cv1): Conv(
          (conv): Conv2d(80, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(80, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (cv2): Conv(
          (conv): Conv2d(80, 80, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(80, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
      )
      (2): Bottleneck(
        (cv1): Conv(
          (conv): Conv2d(80, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(80, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (cv2): Conv(
          (conv): Conv2d(80, 80, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(80, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
      )
      (3): Bottleneck(
        (cv1): Conv(
          (conv): Conv2d(80, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(80, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (cv2): Conv(
          (conv): Conv2d(80, 80, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(80, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
      )
    )
  )
  (3): Conv(
    (conv): Conv2d(160, 320, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    (bn): BatchNorm2d(320, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
    (act): LeakyReLU(negative_slope=0.1, inplace=True)
  )
  (4): BottleneckCSP(
    (cv1): Conv(
      (conv): Conv2d(320, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(160, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
      (act): LeakyReLU(negative_slope=0.1, inplace=True)
    )
    (cv2): Conv2d(320, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (cv3): Conv2d(160, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (cv4): Conv(
      (conv): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(320, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
      (act): LeakyReLU(negative_slope=0.1, inplace=True)
    )
    (bn): BatchNorm2d(320, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
    (act): LeakyReLU(negative_slope=0.1, inplace=True)
    (m): Sequential(
      (0): Bottleneck(
        (cv1): Conv(
          (conv): Conv2d(160, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(160, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (cv2): Conv(
          (conv): Conv2d(160, 160, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(160, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
      )
      (1): Bottleneck(
        (cv1): Conv(
          (conv): Conv2d(160, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(160, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (cv2): Conv(
          (conv): Conv2d(160, 160, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(160, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
      )
      (2): Bottleneck(
        (cv1): Conv(
          (conv): Conv2d(160, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(160, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (cv2): Conv(
          (conv): Conv2d(160, 160, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(160, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
      )
      (3): Bottleneck(
        (cv1): Conv(
          (conv): Conv2d(160, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(160, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (cv2): Conv(
          (conv): Conv2d(160, 160, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(160, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
      )
      (4): Bottleneck(
        (cv1): Conv(
          (conv): Conv2d(160, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(160, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (cv2): Conv(
          (conv): Conv2d(160, 160, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(160, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
      )
      (5): Bottleneck(
        (cv1): Conv(
          (conv): Conv2d(160, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(160, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (cv2): Conv(
          (conv): Conv2d(160, 160, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(160, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
      )
      (6): Bottleneck(
        (cv1): Conv(
          (conv): Conv2d(160, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(160, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (cv2): Conv(
          (conv): Conv2d(160, 160, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(160, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
      )
      (7): Bottleneck(
        (cv1): Conv(
          (conv): Conv2d(160, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(160, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (cv2): Conv(
          (conv): Conv2d(160, 160, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(160, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
      )
      (8): Bottleneck(
        (cv1): Conv(
          (conv): Conv2d(160, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(160, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (cv2): Conv(
          (conv): Conv2d(160, 160, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(160, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
      )
      (9): Bottleneck(
        (cv1): Conv(
          (conv): Conv2d(160, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(160, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (cv2): Conv(
          (conv): Conv2d(160, 160, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(160, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
      )
      (10): Bottleneck(
        (cv1): Conv(
          (conv): Conv2d(160, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(160, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (cv2): Conv(
          (conv): Conv2d(160, 160, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(160, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
      )
      (11): Bottleneck(
        (cv1): Conv(
          (conv): Conv2d(160, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(160, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (cv2): Conv(
          (conv): Conv2d(160, 160, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(160, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
      )
    )
  )
  (5): Conv(
    (conv): Conv2d(320, 640, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    (bn): BatchNorm2d(640, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
    (act): LeakyReLU(negative_slope=0.1, inplace=True)
  )
  (6): BottleneckCSP(
    (cv1): Conv(
      (conv): Conv2d(640, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(320, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
      (act): LeakyReLU(negative_slope=0.1, inplace=True)
    )
    (cv2): Conv2d(640, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (cv3): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (cv4): Conv(
      (conv): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(640, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
      (act): LeakyReLU(negative_slope=0.1, inplace=True)
    )
    (bn): BatchNorm2d(640, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
    (act): LeakyReLU(negative_slope=0.1, inplace=True)
    (m): Sequential(
      (0): Bottleneck(
        (cv1): Conv(
          (conv): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(320, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (cv2): Conv(
          (conv): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(320, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
      )
      (1): Bottleneck(
        (cv1): Conv(
          (conv): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(320, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (cv2): Conv(
          (conv): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(320, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
      )
      (2): Bottleneck(
        (cv1): Conv(
          (conv): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(320, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (cv2): Conv(
          (conv): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(320, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
      )
      (3): Bottleneck(
        (cv1): Conv(
          (conv): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(320, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (cv2): Conv(
          (conv): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(320, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
      )
      (4): Bottleneck(
        (cv1): Conv(
          (conv): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(320, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (cv2): Conv(
          (conv): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(320, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
      )
      (5): Bottleneck(
        (cv1): Conv(
          (conv): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(320, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (cv2): Conv(
          (conv): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(320, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
      )
      (6): Bottleneck(
        (cv1): Conv(
          (conv): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(320, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (cv2): Conv(
          (conv): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(320, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
      )
      (7): Bottleneck(
        (cv1): Conv(
          (conv): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(320, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (cv2): Conv(
          (conv): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(320, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
      )
      (8): Bottleneck(
        (cv1): Conv(
          (conv): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(320, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (cv2): Conv(
          (conv): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(320, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
      )
      (9): Bottleneck(
        (cv1): Conv(
          (conv): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(320, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (cv2): Conv(
          (conv): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(320, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
      )
      (10): Bottleneck(
        (cv1): Conv(
          (conv): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(320, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (cv2): Conv(
          (conv): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(320, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
      )
      (11): Bottleneck(
        (cv1): Conv(
          (conv): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(320, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (cv2): Conv(
          (conv): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(320, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
      )
    )
  )
  (7): Conv(
    (conv): Conv2d(640, 1280, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    (bn): BatchNorm2d(1280, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
    (act): LeakyReLU(negative_slope=0.1, inplace=True)
  )
  (8): SPP(
    (cv1): Conv(
      (conv): Conv2d(1280, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(640, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
      (act): LeakyReLU(negative_slope=0.1, inplace=True)
    )
    (cv2): Conv(
      (conv): Conv2d(2560, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(1280, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
      (act): LeakyReLU(negative_slope=0.1, inplace=True)
    )
    (m): ModuleList(
      (0): MaxPool2d(kernel_size=5, stride=1, padding=2, dilation=1, ceil_mode=False)
      (1): MaxPool2d(kernel_size=9, stride=1, padding=4, dilation=1, ceil_mode=False)
      (2): MaxPool2d(kernel_size=13, stride=1, padding=6, dilation=1, ceil_mode=False)
    )
  )
  (9): BottleneckCSP(
    (cv1): Conv(
      (conv): Conv2d(1280, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(640, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
      (act): LeakyReLU(negative_slope=0.1, inplace=True)
    )
    (cv2): Conv2d(1280, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (cv3): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (cv4): Conv(
      (conv): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(1280, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
      (act): LeakyReLU(negative_slope=0.1, inplace=True)
    )
    (bn): BatchNorm2d(1280, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
    (act): LeakyReLU(negative_slope=0.1, inplace=True)
    (m): Sequential(
      (0): Bottleneck(
        (cv1): Conv(
          (conv): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(640, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (cv2): Conv(
          (conv): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(640, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
      )
      (1): Bottleneck(
        (cv1): Conv(
          (conv): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(640, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (cv2): Conv(
          (conv): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(640, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
      )
      (2): Bottleneck(
        (cv1): Conv(
          (conv): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(640, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (cv2): Conv(
          (conv): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(640, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
      )
      (3): Bottleneck(
        (cv1): Conv(
          (conv): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(640, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (cv2): Conv(
          (conv): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(640, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
      )
    )
  )
  (10): Conv(
    (conv): Conv2d(1280, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn): BatchNorm2d(640, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
    (act): LeakyReLU(negative_slope=0.1, inplace=True)
  )
  (11): Upsample(scale_factor=2.0, mode=nearest)
  (12): Concat()
  (13): BottleneckCSP(
    (cv1): Conv(
      (conv): Conv2d(1280, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(320, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
      (act): LeakyReLU(negative_slope=0.1, inplace=True)
    )
    (cv2): Conv2d(1280, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (cv3): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (cv4): Conv(
      (conv): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(640, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
      (act): LeakyReLU(negative_slope=0.1, inplace=True)
    )
    (bn): BatchNorm2d(640, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
    (act): LeakyReLU(negative_slope=0.1, inplace=True)
    (m): Sequential(
      (0): Bottleneck(
        (cv1): Conv(
          (conv): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(320, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (cv2): Conv(
          (conv): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(320, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
      )
      (1): Bottleneck(
        (cv1): Conv(
          (conv): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(320, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (cv2): Conv(
          (conv): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(320, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
      )
      (2): Bottleneck(
        (cv1): Conv(
          (conv): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(320, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (cv2): Conv(
          (conv): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(320, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
      )
      (3): Bottleneck(
        (cv1): Conv(
          (conv): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(320, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (cv2): Conv(
          (conv): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(320, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
      )
    )
  )
  (14): Conv(
    (conv): Conv2d(640, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn): BatchNorm2d(320, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
    (act): LeakyReLU(negative_slope=0.1, inplace=True)
  )
  (15): Upsample(scale_factor=2.0, mode=nearest)
  (16): Concat()
  (17): BottleneckCSP(
    (cv1): Conv(
      (conv): Conv2d(640, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(160, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
      (act): LeakyReLU(negative_slope=0.1, inplace=True)
    )
    (cv2): Conv2d(640, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (cv3): Conv2d(160, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (cv4): Conv(
      (conv): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(320, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
      (act): LeakyReLU(negative_slope=0.1, inplace=True)
    )
    (bn): BatchNorm2d(320, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
    (act): LeakyReLU(negative_slope=0.1, inplace=True)
    (m): Sequential(
      (0): Bottleneck(
        (cv1): Conv(
          (conv): Conv2d(160, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(160, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (cv2): Conv(
          (conv): Conv2d(160, 160, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(160, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
      )
      (1): Bottleneck(
        (cv1): Conv(
          (conv): Conv2d(160, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(160, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (cv2): Conv(
          (conv): Conv2d(160, 160, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(160, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
      )
      (2): Bottleneck(
        (cv1): Conv(
          (conv): Conv2d(160, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(160, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (cv2): Conv(
          (conv): Conv2d(160, 160, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(160, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
      )
      (3): Bottleneck(
        (cv1): Conv(
          (conv): Conv2d(160, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(160, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (cv2): Conv(
          (conv): Conv2d(160, 160, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(160, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
      )
    )
  )
  (18): Conv2d(320, 255, kernel_size=(1, 1), stride=(1, 1))
  (19): Conv(
    (conv): Conv2d(320, 320, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    (bn): BatchNorm2d(320, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
    (act): LeakyReLU(negative_slope=0.1, inplace=True)
  )
  (20): Concat()
  (21): BottleneckCSP(
    (cv1): Conv(
      (conv): Conv2d(640, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(320, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
      (act): LeakyReLU(negative_slope=0.1, inplace=True)
    )
    (cv2): Conv2d(640, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (cv3): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (cv4): Conv(
      (conv): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(640, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
      (act): LeakyReLU(negative_slope=0.1, inplace=True)
    )
    (bn): BatchNorm2d(640, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
    (act): LeakyReLU(negative_slope=0.1, inplace=True)
    (m): Sequential(
      (0): Bottleneck(
        (cv1): Conv(
          (conv): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(320, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (cv2): Conv(
          (conv): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(320, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
      )
      (1): Bottleneck(
        (cv1): Conv(
          (conv): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(320, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (cv2): Conv(
          (conv): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(320, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
      )
      (2): Bottleneck(
        (cv1): Conv(
          (conv): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(320, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (cv2): Conv(
          (conv): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(320, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
      )
      (3): Bottleneck(
        (cv1): Conv(
          (conv): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(320, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (cv2): Conv(
          (conv): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(320, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
      )
    )
  )
  (22): Conv2d(640, 255, kernel_size=(1, 1), stride=(1, 1))
  (23): Conv(
    (conv): Conv2d(640, 640, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    (bn): BatchNorm2d(640, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
    (act): LeakyReLU(negative_slope=0.1, inplace=True)
  )
  (24): Concat()
  (25): BottleneckCSP(
    (cv1): Conv(
      (conv): Conv2d(1280, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(640, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
      (act): LeakyReLU(negative_slope=0.1, inplace=True)
    )
    (cv2): Conv2d(1280, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (cv3): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (cv4): Conv(
      (conv): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(1280, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
      (act): LeakyReLU(negative_slope=0.1, inplace=True)
    )
    (bn): BatchNorm2d(1280, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
    (act): LeakyReLU(negative_slope=0.1, inplace=True)
    (m): Sequential(
      (0): Bottleneck(
        (cv1): Conv(
          (conv): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(640, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (cv2): Conv(
          (conv): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(640, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
      )
      (1): Bottleneck(
        (cv1): Conv(
          (conv): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(640, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (cv2): Conv(
          (conv): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(640, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
      )
      (2): Bottleneck(
        (cv1): Conv(
          (conv): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(640, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (cv2): Conv(
          (conv): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(640, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
      )
      (3): Bottleneck(
        (cv1): Conv(
          (conv): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(640, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (cv2): Conv(
          (conv): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(640, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True)
          (act): LeakyReLU(negative_slope=0.1, inplace=True)
        )
      )
    )
  )
  (26): Conv2d(1280, 255, kernel_size=(1, 1), stride=(1, 1))
  (27): Detect()
)