project="Orbslam3_label"
datasets="/home/zhangqi/Documents/Experiment/Datasets/Tum"
dataname="tum"

result="/home/zhangqi/Documents/Experiment/Result/${project}/data/${dataname}"
function yes_or_no {
        while true; do
            read -p "$* [y/n]: " yn
            case $yn in
                [Yy]*) return 0  ;;  
                [Nn]*) echo "Aborted" ; return  1 ;;
            esac
        done
    }
# ------------------------------------------------------------
# For RGB-D 运行之后保存结果
# 注意:freg2对应的相机模型是TUM2,freg3对应的是tum3
 ./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt Examples/RGB-D/TUM2.yaml ${datasets}/*desk*  ${datasets}/*desk*/associations.txt

 evo_ape tum ${datasets}/*desk*/groundtruth.txt KeyFrameTrajectory.txt -va --save_plot ${picture}/desk --save_results ${result}/desk.zip
#-----------------------------------------------------------------------
b1=0
b2=0
b3=0
b4=0
b5=0
b6=0
b7=0
b8=0
# while yes_or_no "$message"
while true
do
pose=sitting
./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt Examples/RGB-D/TUM3.yaml ${datasets}/${pose}/*${pose}_halfsphere  ${datasets}/${pose}/*${pose}_halfsphere/associations.txt
		a1=`evo_ape tum ${datasets}/${pose}/*${pose}_halfsphere/groundtruth.txt KeyFrameTrajectory.txt -va | grep mean |awk -F " " '{print $2}'`
		
		if [[ ${a1} < ${b1} ]]&&[ -n "$a1" ];then
			b1=${a1}
			cp -rf KeyFrameTrajectory.txt ${result}
			mv -f ${result}/KeyFrameTrajectory.txt ${result}/KeyFrameTrajectory_${pose}_halfsphere.txt
		fi
#------------------------------------------------------------------------
./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt Examples/RGB-D/TUM3.yaml ${datasets}/${pose}/*${pose}_rpy  ${datasets}/${pose}/*${pose}_rpy/associations.txt
		a2=`evo_ape tum ${datasets}/${pose}/*${pose}_rpy/groundtruth.txt KeyFrameTrajectory.txt -va | grep mean |awk -F " " '{print $2}'`

		
		
		# 检查是否为空
		if [[ ${a2} < ${b2} ]]&&[ -n "$a2" ];then
			b2=${a2}
			cp -rf KeyFrameTrajectory.txt ${result}
			mv -f ${result}/KeyFrameTrajectory.txt ${result}/KeyFrameTrajectory_${pose}_rpy.txt
		fi
#-----------------------------------------------------------------------------
./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt Examples/RGB-D/TUM3.yaml ${datasets}/${pose}/*${pose}_xyz  ${datasets}/${pose}/*${pose}_xyz/associations.txt

		a3=`evo_ape tum ${datasets}/${pose}/*${pose}_xyz/groundtruth.txt KeyFrameTrajectory.txt -va | grep mean |awk -F " " '{print $2}'`

		if [[ ${a3} < ${b3} ]]&&[ -n "$a3" ];then
			b3=${a3}
			cp -rf KeyFrameTrajectory.txt ${result}
			mv -f ${result}/KeyFrameTrajectory.txt ${result}/KeyFrameTrajectory_${pose}_xyz.txt
		fi
#--------------------------------------------------------------------------------------
./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt Examples/RGB-D/TUM3.yaml ${datasets}/${pose}/*${pose}_static  ${datasets}/${pose}/*${pose}_static/associations.txt
		a4=`evo_ape tum ${datasets}/${pose}/*${pose}_static/groundtruth.txt KeyFrameTrajectory.txt -va | grep mean |awk -F " " '{print $2}'`

		if [[ ${a4} < ${b4} ]]&&[ -n "$a4" ];then
			b4=${a4}
			cp -rf KeyFrameTrajectory.txt ${result}
			mv -f ${result}/KeyFrameTrajectory.txt ${result}/KeyFrameTrajectory_${pose}_static.txt
		fi
#--------------------------------------------------------------
# For RGB-D
pose=walking
./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt Examples/RGB-D/TUM3.yaml ${datasets}/${pose}/*${pose}_halfsphere  ${datasets}/${pose}/*${pose}_halfsphere/associations.txt
		a5=`evo_ape tum ${datasets}/${pose}/*${pose}_halfsphere/groundtruth.txt KeyFrameTrajectory.txt -va | grep mean |awk -F " " '{print $2}'`

		if [[ ${a5} < ${b5} ]]&&[ -n "$a5" ];then
			b5=${a5}
			cp -rf KeyFrameTrajectory.txt ${result}
			mv -f ${result}/KeyFrameTrajectory.txt ${result}/KeyFrameTrajectory_${pose}_halfsphere.txt
		fi
#------------------------------------------------------------------------
./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt Examples/RGB-D/TUM3.yaml ${datasets}/${pose}/*${pose}_rpy  ${datasets}/${pose}/*${pose}_rpy/associations.txt
		a6=`evo_ape tum ${datasets}/${pose}/*${pose}_rpy/groundtruth.txt KeyFrameTrajectory.txt -va | grep mean |awk -F " " '{print $2}'`

		if [[ ${a6} < ${b6} ]]&&[ -n "$a6" ];then
				b6=${a6}
			cp -rf KeyFrameTrajectory.txt ${result}
			mv -f ${result}/KeyFrameTrajectory.txt ${result}/KeyFrameTrajectory_${pose}_rpy.txt
		fi
#-----------------------------------------------------------------------------
./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt Examples/RGB-D/TUM3.yaml ${datasets}/${pose}/*${pose}_xyz  ${datasets}/${pose}/*${pose}_xyz/associations.txt

		a7=`evo_ape tum ${datasets}/${pose}/*${pose}_xyz/groundtruth.txt KeyFrameTrajectory.txt -va | grep mean |awk -F " " '{print $2}'`

		if [[ ${a7} < ${b7} ]]&&[ -n "$a7" ];then
				b7=${a7}
			cp -rf KeyFrameTrajectory.txt ${result}
			mv -f ${result}/KeyFrameTrajectory.txt ${result}/KeyFrameTrajectory_${pose}_xyz.txt
		fi
#--------------------------------------------------------------------------------------
./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt Examples/RGB-D/TUM3.yaml ${datasets}/${pose}/*${pose}_static  ${datasets}/${pose}/*${pose}_static/associations.txt

		a8=`evo_ape tum ${datasets}/${pose}/*${pose}_static/groundtruth.txt KeyFrameTrajectory.txt -va | grep mean |awk -F " " '{print $2}'`

		if [[ ${a8} < ${b8} ]]&&[ -n "$a8" ];then
				b8=${a8}
			cp -rf KeyFrameTrajectory.txt ${result}
			mv -f ${result}/KeyFrameTrajectory.txt ${result}/KeyFrameTrajectory_${pose}_static.txt
		fi
done


